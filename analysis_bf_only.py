#!/usr/bin/env python3
"""
Brightfield cell-segmentation results of the 28-well HIB19 / isotype / IVIG experiment
(2026-10-30): caching, QC, figures and a tidy per-well summary table.

Run:   python analysis.py                 (QC-filtered objects, incremental cache)
       python analysis.py --include-small (all segmented objects)
       python analysis.py --no-cache      (force re-reading every Excel file)
       python analysis.py --figures 1,2   (only some figures)

Statistical design choice (applies to every figure and table below):
  The experimental unit is the WELL (duplicate wells = n = 2 per condition), not the cell.
  Tens of thousands of cells from one well are technical sub-samples of the same well,
  so per-cell hypothesis tests would be pseudoreplicated and give vanishing p-values for
  biologically meaningless differences. We therefore
    * summarise each well by one number (count, median, IQR) and show replicate wells
      individually,
    * report per-cell distribution differences only as effect sizes (difference of
      medians vs. the auto control of the same block, bootstrap 95% CI over cells),
      and do not report p-values.
  The bootstrap CI only reflects cell-sampling uncertainty WITHIN a well; it does not
  capture well-to-well variability, which must be judged from the replicate wells.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

# =====================================================================================
# CONFIGURATION – edit here
# =====================================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "."            # folder with <N>_bf.xlsx and *_patemap.xlsx
CACHE_DIR = SCRIPT_DIR / "cache"
FIG_DIR = SCRIPT_DIR / "figures"

FILE_PATTERN = re.compile(r"(\d+)_bf\.xlsx$", re.IGNORECASE)
PLATEMAP_GLOB = "*_patemap.xlsx"
SHEET_NAME = "Measurements"

PIXEL_SIZE_UM = None        # e.g. 0.65 -> lengths in µm, areas in µm²; None = keep pixels
AREA_MIN_PX = 300           # objects with Area < this (in px², before conversion) = debris
HIGH_SMALL_FRAC = 0.15      # fraction_small >= this -> well flagged "high" (†) in figures
BORDERLINE_SMALL_FRAC = 0.10  # fraction_small >= this -> "borderline" (shown on QC panel)
INCLUDE_SMALL_DEFAULT = False  # False = plots use QC-filtered objects (CLI: --include-small)

SEED = 42
MAX_CELLS_PER_WELL_PCA = 20_000
UMAP_CELLS_PER_WELL = 2_000      # UMAP is slow; smaller seeded subsample
VIOLIN_MAX_CELLS = 10_000        # subsample used only for the violin KDE shape
N_BOOT = 1_000
BOOT_MAX_CELLS = 5_000           # cells per well used in the bootstrap (conservative CI)
SPATIAL_BINS = 40
SPATIAL_VMAX = 3.0               # colour-scale cap for relative local density
# =====================================================================================

# ---- experimental design ------------------------------------------------------------
N_WELLS_LAYOUT = 28
ROWS = ["A", "B", "C"]
BLOCK_BSA = "1% BSA"
BLOCK_IVIG = "1% BSA + 10 µg/mL IVIG"
BLOCKS = [BLOCK_BSA, BLOCK_IVIG]
BLOCK_SHORT = {BLOCK_BSA: "BSA", BLOCK_IVIG: "BSA + IVIG"}
REAGENTS = ["HIB19", "iso", "auto"]
REAGENT_LABEL = {"HIB19": "HIB19 (anti-CD19)", "iso": "Isotype control", "auto": "auto (no antibody)"}
REAGENT_ORDER = {r: i for i, r in enumerate(REAGENTS)}
DOSES = [0.5, 1.0, 2.0]
DOSE_UNIT = "µg / 10⁶ cells"

# ---- measurement columns ------------------------------------------------------------
FEATURES = ["Area", "Equivalent Diameter", "Eccentricity", "Circularity", "Perimeter",
            "Solidity", "Bounding Box Width", "Bounding Box Height",
            "Major Axis Length", "Minor Axis Length"]          # 13 columns minus ID/centroids
REQUIRED_COLS = ["Cell ID"] + FEATURES[:4] + ["Centroid X", "Centroid Y"] + FEATURES[4:]
MORPH_FEATURES = ["Area", "Circularity", "Eccentricity", "Solidity"]
AREA_COLS = {"Area"}
LENGTH_COLS = {"Equivalent Diameter", "Perimeter", "Bounding Box Width", "Bounding Box Height",
               "Major Axis Length", "Minor Axis Length", "Centroid X", "Centroid Y"}

# ---- style ---------------------------------------------------------------------------
# Okabe-Ito colour-blind-safe hues; one hue per reagent, lightness encodes dose.
REAGENT_COLOR = {"HIB19": "#0072B2", "iso": "#D55E00", "auto": "#009E73"}
BLOCK_COLOR = {BLOCK_BSA: "#332288", BLOCK_IVIG: "#DDAA33"}
BLOCK_MARKER = {BLOCK_BSA: "o", BLOCK_IVIG: "s"}
DOSE_TINT = {0.0: 0.0, 0.5: 0.60, 1.0: 0.32, 2.0: 0.0}   # mix with white
STATUS_COLOR = {"ok": "#8FA8BF", "borderline": "#E0A030", "high": "#B2342B"}
INK, MUTED, GRID, NODATA, HATCH = "#222222", "#6B6B6B", "#E6E6E6", "#E3E3E3", "#A6A6A6"
SEQ_CMAP = "YlGnBu"
FLAG = "†"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": "#444444", "ytick.color": "#444444", "text.color": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "hatch.color": HATCH, "hatch.linewidth": 0.6,
    "svg.fonttype": "none", "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    "figure.dpi": 100,
})


# =====================================================================================
# helpers
# =====================================================================================
def warn(msg: str) -> None:
    print(f"WARNING: {msg}")


def fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def fmt_wells(wells) -> str:
    """[1,2,3,5,7,8] -> '1–3, 5, 7–8'."""
    wells = sorted(int(w) for w in wells)
    if not wells:
        return "none"
    out, start, prev = [], wells[0], wells[0]
    for w in wells[1:] + [None]:
        if w is not None and w == prev + 1:
            prev = w
            continue
        out.append(f"{start}" if start == prev else f"{start}–{prev}")
        if w is not None:
            start = prev = w
    return ", ".join(out)


def unit(col: str) -> str | None:
    if col in AREA_COLS:
        return "µm²" if PIXEL_SIZE_UM else "px²"
    if col in LENGTH_COLS:
        return "µm" if PIXEL_SIZE_UM else "px"
    return None


def axis_label(col: str) -> str:
    u = unit(col)
    return f"{col} ({u})" if u else f"{col} (unitless)"


def to_physical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pixel units to µm if PIXEL_SIZE_UM is set (QC flag is computed before)."""
    if not PIXEL_SIZE_UM:
        return df
    df = df.copy()
    for c in LENGTH_COLS:
        df[c] = df[c] * PIXEL_SIZE_UM
    for c in AREA_COLS:
        df[c] = df[c] * PIXEL_SIZE_UM ** 2
    return df


def well_color(reagent: str, dose: float):
    base = np.array(mcolors.to_rgb(REAGENT_COLOR[reagent]))
    t = DOSE_TINT.get(float(dose), 0.0)
    return tuple(base + (1 - base) * t)


def text_on(fc) -> str:
    r, g, b = mcolors.to_rgb(fc)
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "white"


def panel_letter(ax, letter: str, x=-0.08, y=1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="bottom", ha="right")


def subsample(a: np.ndarray, n: int, rng) -> np.ndarray:
    return a if len(a) <= n else a[rng.choice(len(a), n, replace=False)]


# =====================================================================================
# metadata
# =====================================================================================
def build_well_meta() -> pd.DataFrame:
    """Hard-coded plate design (checked against the plate-map file, never parsed from it)."""
    doses = [0.5, 0.5, 1.0, 1.0, 2.0, 2.0]
    rec = []
    for w in range(1, N_WELLS_LAYOUT + 1):
        if w <= 24:
            row = "A" if w <= 12 else "B"
            col = (w - 1) % 12 + 1
            block = BLOCK_BSA if w <= 12 else BLOCK_IVIG
            reagent = "HIB19" if col <= 6 else "iso"
            dose = doses[(col - 1) % 6]
            rep = 1 if col % 2 == 1 else 2
        else:
            row, col = "C", w - 24
            block = BLOCK_BSA if w in (25, 26) else BLOCK_IVIG
            reagent, dose = "auto", 0.0
            rep = 1 if w in (25, 27) else 2
        rec.append(dict(well=w, plate_row=row, plate_col=col, block=block,
                        reagent=reagent, dose=dose, replicate=rep))
    return pd.DataFrame(rec)


def _norm_txt(s) -> str:
    return re.sub(r"\s+", "", str(s)).lower().replace("µ", "u")


def check_plate_map(meta: pd.DataFrame) -> None:
    files = sorted(p for p in DATA_DIR.glob(PLATEMAP_GLOB) if not p.name.startswith("~$"))
    if not files:
        warn(f"no plate map ({PLATEMAP_GLOB}) in {DATA_DIR} – design check skipped")
        return
    pm = pd.read_excel(files[0], header=None)
    issues = []

    def cell(r, c):
        return pm.iat[r, c] if r < pm.shape[0] and c < pm.shape[1] else np.nan

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    for m in meta.itertuples():
        ri, ci = ROWS.index(m.plate_row), m.plate_col - 1
        if num(cell(ri, ci)) != m.well:
            issues.append(f"well {m.well}: plate map has {cell(ri, ci)!r} at {m.plate_row}{m.plate_col}")
        if m.reagent != "auto" and num(cell(5 + ri, ci)) != m.dose:
            issues.append(f"well {m.well}: plate-map dose {cell(5 + ri, ci)!r} != {m.dose}")
        if m.reagent == "auto" and _norm_txt(cell(7, ci)) != "auto":
            issues.append(f"well {m.well}: plate map says {cell(7, ci)!r}, expected 'auto'")
    header = " ".join(str(v) for v in pm.iloc[4].tolist()) if pm.shape[0] > 4 else ""
    for token in ("HIB19", "iso"):
        if token.lower() not in header.lower():
            issues.append(f"reagent header {token!r} not found in plate map")
    for r, block in [(5, BLOCK_BSA), (6, BLOCK_IVIG)]:
        if _norm_txt(cell(r, 12)) != _norm_txt(block):
            issues.append(f"row {ROWS[r - 5]} block: plate map {cell(r, 12)!r} vs {block!r}")
    for c, block in [(0, BLOCK_BSA), (2, BLOCK_IVIG)]:
        if _norm_txt(cell(8, c)) != _norm_txt(block):
            issues.append(f"auto block (col {c + 1}): plate map {cell(8, c)!r} vs {block!r}")
    if issues:
        warn(f"plate map {files[0].name} disagrees with hard-coded design:")
        for i in issues:
            print(f"   - {i}")
    else:
        print(f"Plate map {files[0].name}: matches hard-coded design.")


# =====================================================================================
# loading + incremental cache
# =====================================================================================
def _parquet_ok() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


CACHE_FILE = CACHE_DIR / ("all_cells.parquet" if _parquet_ok() else "all_cells.csv")
MANIFEST_FILE = CACHE_DIR / "manifest.json"


def _read_table(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def _write_table(df: pd.DataFrame, p: Path) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    if p.suffix == ".parquet":
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    tmp.replace(p)


def _excel_engine() -> str:
    try:
        import python_calamine  # noqa: F401  (much faster, optional)
        return "calamine"
    except ImportError:
        return "openpyxl"


def discover_well_files() -> dict[int, Path]:
    if not DATA_DIR.is_dir():
        fail(f"data directory not found: {DATA_DIR}")
    found: dict[int, Path] = {}
    for p in sorted(DATA_DIR.iterdir()):
        if p.name.startswith("~$"):          # Excel lock file of an open workbook
            continue
        m = FILE_PATTERN.search(p.name)
        if not m:
            continue
        w = int(m.group(1))
        if not 1 <= w <= N_WELLS_LAYOUT:
            warn(f"{p.name}: well {w} is not in the 1–{N_WELLS_LAYOUT} layout – ignored")
            continue
        if w in found:
            fail(f"two files for well {w}: {found[w].name} and {p.name}")
        found[w] = p
    if not found:
        fail(f"no files matching '<N>_bf.xlsx' in {DATA_DIR}")
    return found


def read_well_file(path: Path, well: int) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, sheet_name=SHEET_NAME, engine=_excel_engine())
    except ValueError as e:
        fail(f"{path.name}: cannot read sheet '{SHEET_NAME}' ({e})")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        fail(f"{path.name}: missing columns {missing}")
    df = df[REQUIRED_COLS].copy()
    df.insert(0, "well", np.int16(well))
    return df


def _sig(p: Path) -> dict:
    st = p.stat()
    return {"file": p.name, "mtime_ns": st.st_mtime_ns, "size": st.st_size}


def load_cells(no_cache: bool) -> tuple[pd.DataFrame, dict]:
    files = discover_well_files()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if MANIFEST_FILE.exists():
        try:
            manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            warn("cache manifest unreadable – rebuilding cache")
    prev_run = manifest.get("last_run")

    cached = None
    if not no_cache and CACHE_FILE.exists() and manifest.get("files"):
        try:
            cached = _read_table(CACHE_FILE)
        except Exception as e:  # corrupt / partial cache
            warn(f"cache {CACHE_FILE.name} unreadable ({e}) – rebuilding")
    old = manifest.get("files", {}) if cached is not None else {}
    cached_wells = set(cached["well"].unique()) if cached is not None else set()

    keep, to_read, changed = [], [], []
    for w, p in files.items():
        o = old.get(str(w))
        if o and o == _sig(p) and w in cached_wells:
            keep.append(w)
        else:
            to_read.append(w)
            if o:
                changed.append(w)
    removed = sorted(int(w) for w in old if int(w) not in files)

    parts = [cached[cached["well"].isin(keep)]] if keep else []
    for i, w in enumerate(to_read, 1):
        t0 = time.time()
        df = read_well_file(files[w], w)
        parts.append(df)
        print(f"  read {files[w].name:<16} {len(df):>7,} rows  ({time.time() - t0:.1f} s)  [{i}/{len(to_read)}]")
    cells = pd.concat(parts, ignore_index=True).sort_values(["well", "Cell ID"], ignore_index=True)

    if to_read or removed or cached is None:
        _write_table(cells, CACHE_FILE)
    manifest = {
        "files": {str(w): _sig(p) for w, p in files.items()},
        "last_run": {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "wells": sorted(files)},
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    info = dict(present=sorted(files), prev_run=prev_run, read=to_read, reused=keep,
                changed=changed, removed=removed, full_rebuild=cached is None)
    return cells, info


def report_completeness(meta: pd.DataFrame, info: dict) -> None:
    present = set(info["present"])
    missing = sorted(set(meta.well) - present)
    print("\n=== Data completeness =====================================================")
    print(f"Layout wells: {len(meta)} | present: {len(present)} | missing: {len(missing)}")
    print(f"  present: {fmt_wells(present)}")
    print(f"  missing: {fmt_wells(missing)}")
    prev = info["prev_run"]
    if prev:
        before = set(prev["wells"])
        print(f"Since previous run ({prev['time']}): added {fmt_wells(present - before)}; "
              f"removed {fmt_wells(before - present)}; file changed {fmt_wells(info['changed'])}")
    else:
        print("Since previous run: no previous run recorded.")
    src = "full rebuild" if info["full_rebuild"] else "incremental"
    print(f"Cache ({src}): reused {len(info['reused'])} wells, read {len(info['read'])} files"
          + (f", dropped {fmt_wells(info['removed'])}" if info["removed"] else ""))
    m = meta.assign(present=meta.well.isin(present))
    print(f"\n{'block':<24}{'reagent':<8}{'dose':>5}  present/expected  wells present")
    for (b, r, d), g in m.groupby(["block", "reagent", "dose"], sort=False):
        wp = g.well[g.present].tolist()
        miss = g.well[~g.present].tolist()
        tail = fmt_wells(wp) + (f"   (missing {fmt_wells(miss)})" if miss else "")
        print(f"{b:<24}{r:<8}{d:>5g}  {len(wp)}/{len(g):<15} {tail}")
    print("===========================================================================\n")


# =====================================================================================
# context: well table, selection
# =====================================================================================
@dataclass
class Ctx:
    cells_all: pd.DataFrame          # all objects, pixel units (QC uses this)
    cells: pd.DataFrame              # plotted selection, physical units if configured
    wells: pd.DataFrame              # one row per layout well (present or not)
    include_small: bool
    by_well: dict = field(default_factory=dict)
    pca: dict | None = None

    @property
    def count_col(self) -> str:
        return "n_total" if self.include_small else "n_qc"

    @property
    def objects_label(self) -> str:
        return ("all segmented objects" if self.include_small
                else f"QC-filtered objects (Area ≥ {AREA_MIN_PX} px²)")

    @property
    def missing(self) -> list[int]:
        return self.wells.well[~self.wells.present].tolist()

    def values(self, feature: str, well: int):
        g = self.by_well.get(well)
        return None if g is None or len(g) == 0 else g[feature].to_numpy(float)


def build_well_table(meta: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    g = cells.groupby("well")
    t = pd.DataFrame({"n_total": g.size(), "n_small": g["is_small"].sum()})
    t["n_qc"] = t.n_total - t.n_small
    t["fraction_small"] = t.n_small / t.n_total
    t["qc_status"] = np.select([t.fraction_small >= HIGH_SMALL_FRAC,
                                t.fraction_small >= BORDERLINE_SMALL_FRAC],
                               ["high", "borderline"], "ok")
    w = meta.merge(t, left_on="well", right_index=True, how="left")
    w["present"] = w.n_total.notna()
    w["qc_status"] = w.qc_status.fillna("no data")
    return w


def flag(row) -> str:
    return FLAG if row.qc_status == "high" else ""


def footer(fig, ctx: Ctx, extra: str = "", y: float = -0.01) -> None:
    lines = [f"Objects: {ctx.objects_label}.  Wells without data: {fmt_wells(ctx.missing)}.  "
             f"{FLAG} fraction of objects with Area < {AREA_MIN_PX} px² ≥ {HIGH_SMALL_FRAC:.0%} "
             f"(possible debris / segmentation noise)."]
    if extra:
        lines.append(extra)
    fig.text(0.01, y, "\n".join(lines), fontsize=6, color=MUTED, ha="left", va="top")


def save(fig, name: str, ctx: Ctx, suffix: bool = True) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = name + ("_allobjects" if (suffix and ctx.include_small) else "")
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  saved figures/{stem}.png/.svg")


# =====================================================================================
# summary table
# =====================================================================================
def export_summary(ctx: Ctx) -> pd.DataFrame:
    rows = []
    for w in ctx.wells.itertuples():
        r = {}
        for f in MORPH_FEATURES:
            v = ctx.values(f, w.well)
            key = f.replace(" ", "_")
            if v is None:
                r.update({f"{key}_median": np.nan, f"{key}_q25": np.nan,
                          f"{key}_q75": np.nan, f"{key}_IQR": np.nan})
            else:
                q25, med, q75 = np.percentile(v, [25, 50, 75])
                r.update({f"{key}_median": med, f"{key}_q25": q25,
                          f"{key}_q75": q75, f"{key}_IQR": q75 - q25})
        rows.append(r)
    out = pd.concat([ctx.wells.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["objects_used_for_morphology"] = "all" if ctx.include_small else "qc_filtered"
    out["area_unit"] = unit("Area")
    cols = ["well", "plate_row", "plate_col", "block", "reagent", "dose", "replicate", "present",
            "n_total", "n_small", "n_qc", "fraction_small", "qc_status"]
    out = out[cols + [c for c in out.columns if c not in cols]]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    name = "well_summary" + ("_allobjects" if ctx.include_small else "") + ".csv"
    out.to_csv(FIG_DIR / name, index=False, float_format="%.6g")
    print(f"  saved figures/{name}")
    return out


# =====================================================================================
# Figure 1 – plate heatmap
# =====================================================================================
def draw_plate(ax, wells: pd.DataFrame, value_col: str, fmt, title: str, vmin=None):
    cm = plt.get_cmap(SEQ_CMAP)
    vals = wells.loc[wells.present, value_col].dropna()
    lo = vals.min() if vmin is None else vmin
    norm = mcolors.Normalize(lo, vals.max() if len(vals) else 1)
    for w in wells.itertuples():
        x, y = w.plate_col - 1, ROWS.index(w.plate_row)
        v = getattr(w, value_col)
        has = bool(w.present) and pd.notna(v)
        fc = cm(norm(v)) if has else NODATA
        ax.add_patch(Rectangle((x + .05, y + .05), .9, .9, facecolor=fc, lw=0,
                               edgecolor=HATCH if not has else "none",
                               hatch=None if has else "////"))
        if w.qc_status == "high":
            ax.add_patch(Rectangle((x + .05, y + .05), .9, .9, fill=False, edgecolor=INK, lw=1.6))
        tc = text_on(fc)
        ax.text(x + .12, y + .2, f"{w.well}{flag(w)}", fontsize=6, ha="left", va="center", color=tc)
        ax.text(x + .5, y + .54, fmt(v) if has else "no data", ha="center", va="center",
                fontsize=7 if has else 5.5, color=tc if has else MUTED,
                fontweight="bold" if has else "normal")
        sub = f"{w.dose:g} µg" if w.reagent != "auto" else BLOCK_SHORT[w.block]
        ax.text(x + .5, y + .83, sub, ha="center", va="center", fontsize=5, color=tc)
    ax.set_xlim(0, 12)
    ax.set_ylim(3, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(12) + .5, labels=[str(i) for i in range(1, 13)])
    ax.xaxis.tick_top()
    ax.set_yticks([.5, 1.5, 2.5], labels=ROWS)
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    kw = dict(fontsize=6.5, color=MUTED, clip_on=False)
    ax.text(3, -0.45, REAGENT_LABEL["HIB19"], ha="center", va="bottom", **kw)
    ax.text(9, -0.45, REAGENT_LABEL["iso"], ha="center", va="bottom", **kw)
    ax.text(12.15, .5, BLOCK_SHORT[BLOCK_BSA], ha="left", va="center", **kw)
    ax.text(12.15, 1.5, BLOCK_SHORT[BLOCK_IVIG], ha="left", va="center", **kw)
    ax.text(4.15, 2.5, REAGENT_LABEL["auto"], ha="left", va="center", **kw)
    ax.set_title(title, pad=22)
    return ScalarMappable(norm=norm, cmap=cm)


def _fmt_count(v):
    return f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}"


def _fmt_area(v):
    return f"{v:.0f}" if v >= 100 else f"{v:.3g}"


def _add_median_area(ctx: Ctx) -> pd.DataFrame:
    w = ctx.wells.copy()
    w["median_area"] = [np.median(v) if (v := ctx.values("Area", x)) is not None else np.nan
                        for x in w.well]
    return w


def fig_plate_heatmap(ctx: Ctx) -> None:
    w = _add_median_area(ctx)
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.6))
    m = draw_plate(axes[0], w, ctx.count_col, _fmt_count, "a  Cell count per well", vmin=0)
    fig.colorbar(m, ax=axes[0], fraction=0.02, pad=0.14, label="Cell count\n(objects / well)")
    m = draw_plate(axes[1], w, "median_area", _fmt_area, "b  Median cell area per well")
    fig.colorbar(m, ax=axes[1], fraction=0.02, pad=0.14, label=f"Median\n{axis_label('Area')}")
    fig.subplots_adjust(hspace=0.3)
    footer(fig, ctx, "Thick outline + † = high debris fraction. Count colour scale starts at 0; "
                     "area colour scale spans the observed range.")
    save(fig, "fig1_plate_heatmap", ctx)


# =====================================================================================
# Figure 2 – dose-response of cell count
# =====================================================================================
X_AUTO = -2.0
R_OFFSET = {"HIB19": -0.09, "iso": 0.09, "auto": 0.0}


def _xpos(dose: float) -> float:
    return X_AUTO if dose == 0 else float(np.log2(dose))


def dose_table(ctx: Ctx, normalized: bool):
    d = ctx.wells[ctx.wells.present].copy()
    d["y"] = d[ctx.count_col].astype(float)
    base = {}
    for b in BLOCKS:
        a = d[(d.block == b) & (d.reagent == "auto")]
        mean = a.y.mean() if len(a) else np.nan
        base[b] = (mean if mean and mean > 0 else np.nan, a.well.tolist())
    if normalized:
        d["y"] = d.y / d.block.map(lambda b: base[b][0])
    return d, base


def draw_dose_response(ax, ctx: Ctx, block: str, normalized: bool, label_wells=True):
    d, base = dose_table(ctx, normalized)
    sub = d[d.block == block]
    bmean, bwells = base[block]
    k = len(bwells)
    btxt = (f"baseline = mean of n = {k} auto well{'s' if k != 1 else ''} ({fmt_wells(bwells)})"
            if k else "no auto well present in this block")
    ax.set_title(BLOCK_LABEL_LONG(block), loc="left", pad=14)
    ax.text(0, 1.01, btxt, transform=ax.transAxes, fontsize=6.5, color=MUTED, va="bottom")
    if normalized and np.isnan(bmean):
        ax.text(.5, .5, "cannot normalise:\nno auto well in this block", transform=ax.transAxes,
                ha="center", va="center", color=MUTED)
    else:
        ref = 1.0 if normalized else bmean
        if not np.isnan(ref):
            ax.axhline(ref, color=MUTED, ls=":", lw=1, zorder=1)
        for reagent in REAGENTS:
            r = sub[sub.reagent == reagent]
            if r.empty:
                continue
            c = REAGENT_COLOR[reagent]
            xs = np.array([_xpos(x) for x in r.dose]) + R_OFFSET[reagent] + (r.replicate - 1.5) * 0.07
            ax.scatter(xs, r.y, s=26, color=[well_color(reagent, x) for x in r.dose],
                       edgecolor=c, lw=0.9, zorder=3, marker=BLOCK_MARKER[block])
            if reagent != "auto":
                means = r.groupby("dose").y.mean().reindex(DOSES)
                mx = np.array([_xpos(x) for x in DOSES]) + R_OFFSET[reagent]
                ax.plot(mx, means.values, color=c, lw=2, zorder=2, solid_capstyle="round")
                ax.scatter(mx, means.values, marker="_", s=110, color=c, lw=2, zorder=2)
            if label_wells:
                for xx, yy, row in zip(xs, r.y, r.itertuples()):
                    left = reagent == "HIB19" or (reagent == "auto" and row.replicate == 1)
                    ax.annotate(f"{row.well}{flag(row)}", (xx, yy), fontsize=5.5, color=MUTED,
                                xytext=(-5 if left else 5, 0), textcoords="offset points",
                                ha="right" if left else "left", va="center")
        # data-driven "no data" notes for conditions in the layout without any well
        lay = ctx.wells[(ctx.wells.block == block) & (ctx.wells.reagent != "auto")]
        notes = []
        for reagent in ["HIB19", "iso"]:
            gone = [x for x in DOSES if not ((sub.reagent == reagent) & (sub.dose == x)).any()
                    and ((lay.reagent == reagent) & (lay.dose == x)).any()]
            if gone:
                notes.append(f"no data: {reagent} {', '.join(f'{x:g}' for x in gone)} µg")
        if notes:
            ax.text(0.98, 0.03, "\n".join(notes), transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=6, color=MUTED)
    ax.axvline(-1.5, color=GRID, lw=0.8, zorder=0)
    ax.set_xticks([X_AUTO, -1, 0, 1], labels=["auto", "0.5", "1", "2"])
    ax.set_xlim(-2.5, 1.5)
    ax.set_xlabel(f"Antibody dose ({DOSE_UNIT})")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    return d.y.max()


def set_dose_ylim(axes, ymax):
    """Shared y range from 0 to the largest value of ANY block (data-driven)."""
    top = (ymax if np.isfinite(ymax) else 1.0) * 1.1
    for ax in axes:
        ax.set_ylim(0, top)


def BLOCK_LABEL_LONG(block: str) -> str:
    return f"Block: {block}"


def _dose_legend(fig, y=1.02):
    h = [Line2D([], [], color=REAGENT_COLOR[r], marker="o", lw=2 if r != "auto" else 0,
                markersize=5, label=REAGENT_LABEL[r]) for r in REAGENTS]
    h.append(Line2D([], [], color=MUTED, ls=":", lw=1, label="auto mean"))
    h.append(Line2D([], [], color=MUTED, marker="o", lw=0, mfc="white", label="BSA (circle)"))
    h.append(Line2D([], [], color=MUTED, marker="s", lw=0, mfc="white", label="BSA + IVIG (square)"))
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, y))


def fig_dose_response(ctx: Ctx) -> None:
    for normalized in (True, False):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
        ymax = max(draw_dose_response(ax, ctx, b, normalized) for ax, b in zip(axes, BLOCKS))
        set_dose_ylim(axes, ymax)
        axes[0].set_ylabel("Cell count / mean of auto wells (same block)" if normalized
                           else "Cell count (objects / well)")
        _dose_legend(fig)
        fig.tight_layout()
        footer(fig, ctx, "Dots: individual wells (number next to dot); line and bar: mean of the "
                         "replicate wells per dose. Light→dark fill = increasing dose.")
        save(fig, "fig2_dose_response_" + ("normalized" if normalized else "raw"), ctx)


# =====================================================================================
# Figure 3 – morphology distributions (+ effect sizes)
# =====================================================================================
def block_positions(wells: pd.DataFrame, block: str) -> pd.DataFrame:
    sub = wells[wells.block == block].copy()
    sub["ro"] = sub.reagent.map(REAGENT_ORDER)
    sub = sub.sort_values(["ro", "dose", "replicate"])
    pos, x, prev = [], 0.0, None
    for r in sub.itertuples():
        if prev is not None and r.reagent != prev:
            x += 0.7
        pos.append(x)
        x += 1
        prev = r.reagent
    sub["xpos"] = pos
    return sub


def _display(feature, v):
    # Circularity > 1 is a pixel-perimeter artefact; clip for display only.
    return np.minimum(v, 1.0) if feature == "Circularity" else v


def feature_ylim(ctx: Ctx, feature: str):
    v = _display(feature, ctx.cells[feature].to_numpy(float))
    lo, hi = np.percentile(v, [0.5, 99.5])
    pad = (hi - lo) * 0.04
    return lo - pad, hi + pad


def _group_labels(ax, sub: pd.DataFrame, y=-0.30):
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for reagent, g in sub.groupby("reagent", sort=False):
        ax.text(g.xpos.mean(), y, reagent if reagent != "auto" else "auto", transform=tr,
                ha="center", va="top", fontsize=7, color=INK)


def draw_violins(ax, ctx: Ctx, block: str, feature: str, ylim, label_x=True, rng=None):
    rng = rng or np.random.default_rng(SEED)
    sub = block_positions(ctx.wells, block)
    for r in sub.itertuples():
        v = ctx.values(feature, r.well)
        if v is None:
            ax.add_patch(Rectangle((r.xpos - .4, ylim[0]), .8, ylim[1] - ylim[0], facecolor="#F1F1F1",
                                   edgecolor=HATCH, hatch="////", lw=0, zorder=0))
            ax.text(r.xpos, np.mean(ylim), "no data", rotation=90, ha="center", va="center",
                    fontsize=5.5, color=MUTED)
            continue
        v = _display(feature, v)
        kv = subsample(v, VIOLIN_MAX_CELLS, rng)
        if len(kv) >= 10 and np.ptp(kv) > 0:
            parts = ax.violinplot(kv, positions=[r.xpos], widths=.85, showextrema=False, points=200)
            for b in parts["bodies"]:
                b.set_facecolor(well_color(r.reagent, r.dose))
                b.set_edgecolor(REAGENT_COLOR[r.reagent])
                b.set_linewidth(0.5)
                b.set_alpha(1)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(r.xpos, q1, q3, color=INK, lw=1.4, zorder=3)
        ax.scatter([r.xpos], [med], s=9, color="white", edgecolor=INK, lw=.7, zorder=4)
        if r.qc_status == "high":
            ax.text(r.xpos, ylim[1], FLAG, ha="center", va="top", fontsize=8, color=INK)
    ax.set_ylim(ylim)
    ax.set_xlim(-0.6, sub.xpos.max() + 0.6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if label_x:
        ax.set_xticks(sub.xpos, labels=[f"{r.well}\n{'–' if r.reagent == 'auto' else f'{r.dose:g}'}"
                                        for r in sub.itertuples()])
        _group_labels(ax, sub)
    else:
        ax.set_xticks(sub.xpos, labels=[])
    ax.tick_params(axis="x", length=0)
    return sub


def _dose_color_legend(fig, y=-0.02, ncol=7):
    h = []
    for reagent in ["HIB19", "iso"]:
        for d in DOSES:
            h.append(Patch(facecolor=well_color(reagent, d), edgecolor=REAGENT_COLOR[reagent],
                           label=f"{reagent} {d:g} µg"))
    h.append(Patch(facecolor=well_color("auto", 0), edgecolor=REAGENT_COLOR["auto"], label="auto"))
    fig.legend(handles=h, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y))


def fig_morphology(ctx: Ctx) -> None:
    widths = [len(block_positions(ctx.wells, b)) + 1.4 for b in BLOCKS]
    fig, axes = plt.subplots(len(MORPH_FEATURES), 2, figsize=(10, 8.6), sharey="row",
                             gridspec_kw=dict(width_ratios=widths, hspace=0.12, wspace=0.06))
    rng = np.random.default_rng(SEED)
    last = len(MORPH_FEATURES) - 1
    for i, f in enumerate(MORPH_FEATURES):
        yl = feature_ylim(ctx, f)
        for j, b in enumerate(BLOCKS):
            ax = axes[i, j]
            draw_violins(ax, ctx, b, f, yl, label_x=(i == last), rng=rng)
            if i == 0:
                ax.set_title(f"Block: {b}", pad=10)
            if j == 0:
                ax.set_ylabel(axis_label(f))
    axes[last, 0].text(-0.07, -0.04, "well\ndose", transform=axes[last, 0].transAxes, ha="right",
                       va="top", fontsize=6, color=MUTED)
    _dose_color_legend(fig, y=0.04)
    footer(fig, ctx, f"Violins: kernel density of up to {VIOLIN_MAX_CELLS:,} randomly sampled cells per "
                     "well; white dot = median, black bar = IQR (all cells). y-axis limited to the "
                     "0.5–99.5th percentile of all cells; Circularity clipped at 1 for display.",
           y=0.005)
    save(fig, "fig3_morphology_violins", ctx)
    fig_effect_sizes(ctx)


def _boot_medians(v: np.ndarray, rng) -> np.ndarray:
    v = subsample(v, BOOT_MAX_CELLS, rng)
    out = np.empty(N_BOOT)
    for s in range(0, N_BOOT, 250):
        n = min(250, N_BOOT - s)
        out[s:s + n] = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return out


def compute_effect_sizes(ctx: Ctx) -> pd.DataFrame:
    """Median(well) - mean of auto-well medians (same block), bootstrap 95% CI over cells.

    The baseline is the mean of per-well medians of the auto wells (well = unit), and the
    bootstrap resamples cells within each well independently. Using at most BOOT_MAX_CELLS
    cells per well makes the CI wider (conservative). The CI reflects only within-well
    sampling noise, NOT well-to-well variability; compare replicate wells for that.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for f in MORPH_FEATURES:
        for b in BLOCKS:
            autos = ctx.wells[(ctx.wells.block == b) & (ctx.wells.reagent == "auto")].well
            av = [ctx.values(f, a) for a in autos]
            av = [_display(f, a) for a in av if a is not None]
            if not av:
                continue
            base = np.mean([np.median(a) for a in av])
            base_boot = np.mean([_boot_medians(a, rng) for a in av], axis=0)
            for w in ctx.wells[ctx.wells.block == b].itertuples():
                v = ctx.values(f, w.well)
                if v is None:
                    continue
                v = _display(f, v)
                diff = np.median(v) - base
                bd = _boot_medians(v, rng) - base_boot
                # The bootstrap runs on a <= BOOT_MAX_CELLS subsample, so its centre can differ
                # slightly from the full-data estimate; shift it onto the full-data estimate
                # (CI width from the subsample = conservative, location from all cells).
                bd = bd - np.median(bd) + diff
                lo, hi = np.percentile(bd, [2.5, 97.5])
                rows.append(dict(feature=f, well=w.well, block=b, reagent=w.reagent, dose=w.dose,
                                 n_auto_wells=len(av), median_diff=diff, ci_low=lo, ci_high=hi,
                                 rel_diff=diff / base if base else np.nan))
    return pd.DataFrame(rows)


def fig_effect_sizes(ctx: Ctx) -> None:
    es = compute_effect_sizes(ctx)
    name = "morph_effect_sizes" + ("_allobjects" if ctx.include_small else "") + ".csv"
    if es.empty:
        print("  no auto well present in any block – effect sizes (fig3b) skipped")
        return
    es.to_csv(FIG_DIR / name, index=False, float_format="%.6g")
    print(f"  saved figures/{name}")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    pos_blocks, off = {}, 0.0
    for b in BLOCKS:
        p = block_positions(ctx.wells, b)
        p["x"] = p.xpos + off
        pos_blocks[b] = p
        off = p.x.max() + 2.0
    allpos = pd.concat(pos_blocks.values())
    for ax, f in zip(axes.flat, MORPH_FEATURES):
        e = es[es.feature == f].merge(allpos[["well", "x", "replicate", "qc_status"]], on="well")
        ax.axhline(0, color=MUTED, lw=0.8)
        for r in e.itertuples():
            ax.errorbar(r.x, r.median_diff, yerr=[[r.median_diff - r.ci_low], [r.ci_high - r.median_diff]],
                        fmt="none", ecolor=INK, elinewidth=0.9, capsize=0, zorder=2)
            ax.scatter(r.x, r.median_diff, s=28, marker=BLOCK_MARKER[r.block],
                       color=well_color(r.reagent, r.dose), edgecolor=REAGENT_COLOR[r.reagent],
                       lw=0.9, zorder=3)
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        for b, p in pos_blocks.items():
            nb = e[e.block == b].n_auto_wells
            tag = (f"vs. {int(nb.iloc[0])} auto well{'s' if nb.iloc[0] != 1 else ''}" if len(nb)
                   else "no auto baseline")
            ax.text(p.x.mean(), 1.0, f"{BLOCK_SHORT[b]}  ({tag})", transform=tr, ha="center",
                    va="bottom", fontsize=6.5, color=MUTED)
            for r in p.itertuples():
                if not (e.well == r.well).any():
                    ax.text(r.x, 0.03, "n/a", transform=tr, ha="center", fontsize=5, color=MUTED,
                            rotation=90)
        ax.set_xticks(allpos.x, labels=[f"{r.well}{flag(r)}" for r in allpos.itertuples()], fontsize=6)
        ax.set_xlim(allpos.x.min() - .8, allpos.x.max() + .8)
        u = unit(f)
        ax.set_ylabel(f"Δ median {f}" + (f" ({u})" if u else "") + "\nvs. auto (same block)")
        ax.set_title(f, pad=12)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Well")
    fig.tight_layout(h_pad=2.2)
    _dose_color_legend(fig, y=0.0)
    footer(fig, ctx, "Point: median of the well minus mean of the auto-well medians of the same block. "
                     f"Error bar: bootstrap 95% CI over cells ({N_BOOT} resamples, ≤{BOOT_MAX_CELLS:,} "
                     "cells/well) – within-well sampling noise only;\nwell-to-well variability is shown by "
                     "the replicate wells. No p-values reported (experimental unit = well, n = 2).",
           y=-0.06)
    save(fig, "fig3b_morphology_effect_sizes", ctx)


# =====================================================================================
# Figure 4 – PCA / UMAP fingerprint
# =====================================================================================
def compute_pca(ctx: Ctx) -> dict:
    if ctx.pca is not None:
        return ctx.pca
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(SEED)
    parts = []
    for w, g in ctx.by_well.items():
        if len(g) == 0:
            continue
        idx = np.sort(rng.choice(len(g), min(len(g), MAX_CELLS_PER_WELL_PCA), replace=False))
        parts.append(g.iloc[idx][["well"] + FEATURES])
    s = pd.concat(parts, ignore_index=True)
    # Note: 7 of the 10 features measure size, so PC1 is expected to be mostly "cell size".
    scaler = StandardScaler().fit(s[FEATURES].to_numpy(float))
    X = scaler.transform(s[FEATURES].to_numpy(float))
    pca = PCA(n_components=min(5, len(FEATURES)), random_state=SEED).fit(X)
    sc = pca.transform(X)
    for i in range(3):
        s[f"PC{i + 1}"] = sc[:, i]
    meta = ctx.wells.set_index("well")
    for c in ["block", "reagent", "dose", "replicate", "qc_status"]:
        s[c] = s.well.map(meta[c])
    cen = s.groupby("well")[["PC1", "PC2"]].median().join(meta[["block", "reagent", "dose", "qc_status"]])
    ctx.pca = dict(sample=s, X=X, pca=pca, centroids=cen.reset_index())
    cen.reset_index().to_csv(FIG_DIR / ("pca_well_centroids" + ("_allobjects" if ctx.include_small else "")
                                        + ".csv"), index=False, float_format="%.6g")
    return ctx.pca


def _pc_label(pca, i):
    return f"PC{i + 1} ({pca.explained_variance_ratio_[i]:.0%} of variance)"


def _pc_limits(s):
    return (np.percentile(s.PC1, [0.2, 99.8]), np.percentile(s.PC2, [0.2, 99.8]))


def draw_pca_centroids(ax, ctx: Ctx, background=False, label=True):
    p = compute_pca(ctx)
    s, cen = p["sample"], p["centroids"]
    xl, yl = _pc_limits(s)
    if background:
        ax.hexbin(s.PC1, s.PC2, gridsize=70, bins="log", cmap="Greys", mincnt=1, linewidths=0,
                  extent=(*xl, *yl), alpha=0.55, rasterized=True)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for r in cen.itertuples():
        ax.scatter(r.PC1, r.PC2, s=42, marker=BLOCK_MARKER[r.block], color=well_color(r.reagent, r.dose),
                   edgecolor=REAGENT_COLOR[r.reagent], lw=1.1, zorder=3)
        if label:
            ax.annotate(f"{r.well}{FLAG if r.qc_status == 'high' else ''}", (r.PC1, r.PC2), fontsize=5.5,
                        color=INK, xytext=(3, 3), textcoords="offset points", zorder=4)
    ax.set_xlabel(_pc_label(p["pca"], 0))
    ax.set_ylabel(_pc_label(p["pca"], 1))
    # zoom on the well medians (they sit close together relative to the cell cloud)
    px = (cen.PC1.max() - cen.PC1.min()) * 0.15 + 0.2
    py = (cen.PC2.max() - cen.PC2.min()) * 0.15 + 0.2
    ax.set_xlim(cen.PC1.min() - px, cen.PC1.max() + px)
    ax.set_ylim(cen.PC2.min() - py, cen.PC2.max() + py)
    return xl, yl


def _hdr_contours(ax, x, y, xl, yl, color, levels=(0.5, 0.9), bins=120, sigma=2.0):
    """Highest-density contours enclosing `levels` of the cells (smoothed 2D histogram)."""
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[xl, yl])
    H = gaussian_filter(H, sigma)
    flat = np.sort(H.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    th = sorted({flat[min(np.searchsorted(cum, lv), len(flat) - 1)] for lv in levels})
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    ax.contour(xc, yc, H.T, levels=th, colors=[color], linewidths=[0.9, 1.8][:len(th)],
               linestyles=["--", "-"][:len(th)])


def _scatter_by(ax, s, key, colors, title, xl, yl, rng=None):
    """Grey hexbin of all sampled cells + per-group 50% (solid) / 90% (dashed) density contours.

    (A plain scatter of ~400k cells overplots into an unreadable blend.)"""
    ax.hexbin(s.PC1, s.PC2, gridsize=70, bins="log", cmap="Greys", mincnt=1, linewidths=0,
              extent=(*xl, *yl), alpha=0.35, rasterized=True)
    for k, c in colors.items():
        g = s[s[key] == k]
        if len(g) > 100:
            _hdr_contours(ax, g.PC1.to_numpy(), g.PC2.to_numpy(), xl, yl, c)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_title(title)


def _contour_key():
    return [Line2D([], [], lw=1.8, color=MUTED, label="50% of cells"),
            Line2D([], [], lw=0.9, ls="--", color=MUTED, label="90% of cells")]


def fig_pca(ctx: Ctx) -> None:
    p = compute_pca(ctx)
    s, pca = p["sample"], p["pca"]
    rng = np.random.default_rng(SEED)
    xl, yl = _pc_limits(s)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.2))
    _scatter_by(axes[0, 0], s, "reagent", REAGENT_COLOR, "a  Cell density by reagent", xl, yl, rng)
    axes[0, 0].legend(handles=[Line2D([], [], lw=1.8, color=REAGENT_COLOR[r], label=REAGENT_LABEL[r])
                               for r in REAGENTS] + _contour_key(), loc="lower right")
    _scatter_by(axes[0, 1], s, "block", BLOCK_COLOR, "b  Cell density by block", xl, yl, rng)
    axes[0, 1].legend(handles=[Line2D([], [], lw=1.8, color=BLOCK_COLOR[b], label=b)
                               for b in BLOCKS] + _contour_key(), loc="lower right")
    for ax in axes[0]:
        ax.set_xlabel(_pc_label(pca, 0))
        ax.set_ylabel(_pc_label(pca, 1))
    draw_pca_centroids(axes[1, 0], ctx)
    axes[1, 0].set_title("c  Per-well median position (zoomed)")
    axes[1, 0].legend(handles=[Line2D([], [], marker=BLOCK_MARKER[b], lw=0, mfc="white", mec=MUTED,
                                      label=BLOCK_SHORT[b]) for b in BLOCKS], loc="upper right")
    ax = axes[1, 1]
    L = pca.components_[:3].T
    lim = np.abs(L).max()
    im = ax.imshow(L, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(L):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(v) > 0.6 * lim else INK)
    ax.set_xticks(range(3), labels=[f"PC{i + 1}\n({pca.explained_variance_ratio_[i]:.0%})" for i in range(3)])
    ax.set_yticks(range(len(FEATURES)), labels=FEATURES)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("d  Loadings (standardised features)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Loading")
    fig.tight_layout()
    _dose_color_legend(fig, y=0.0)
    footer(fig, ctx, f"PCA on {len(FEATURES)} z-scored features ({', '.join(FEATURES)}); up to "
                     f"{MAX_CELLS_PER_WELL_PCA:,} randomly sampled cells per well (seed {SEED}), "
                     f"{len(s):,} cells in total. Axes limited to the 0.2–99.8th percentile.", y=-0.04)
    save(fig, "fig4_pca", ctx)
    fig_umap(ctx)


def fig_umap(ctx: Ctx) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import umap
    except Exception as e:  # ImportError or broken numba install
        print(f"  umap-learn not available ({type(e).__name__}) – UMAP skipped")
        return
    p = compute_pca(ctx)
    s, X = p["sample"], p["X"]
    rng = np.random.default_rng(SEED)
    idx = np.concatenate([subsample(np.flatnonzero(s.well.to_numpy() == w), UMAP_CELLS_PER_WELL, rng)
                          for w in s.well.unique()])
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=SEED).fit_transform(X[idx])
    print(f"  UMAP on {len(idx):,} cells ({time.time() - t0:.0f} s)")
    u = s.iloc[idx].assign(U1=emb[:, 0], U2=emb[:, 1])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, key, cmap, title, labels in [
        (axes[0], "reagent", REAGENT_COLOR, "a  UMAP coloured by reagent", REAGENT_LABEL),
        (axes[1], "block", BLOCK_COLOR, "b  UMAP coloured by block", {b: b for b in BLOCKS})]:
        o = rng.permutation(len(u))
        ax.scatter(u.U1.to_numpy()[o], u.U2.to_numpy()[o], s=1, alpha=0.25, lw=0,
                   c=list(u[key].map(cmap).to_numpy()[o]), rasterized=True)
        ax.legend(handles=[Line2D([], [], marker="o", lw=0, color=c, label=labels[k]) for k, c in cmap.items()],
                  loc="best")
        ax.set_xlabel("UMAP 1 (arbitrary units)")
        ax.set_ylabel("UMAP 2 (arbitrary units)")
        ax.set_title(title)
    fig.tight_layout()
    footer(fig, ctx, f"UMAP (n_neighbors = 30, min_dist = 0.3, seed {SEED}) on the z-scored features of up to "
                     f"{UMAP_CELLS_PER_WELL:,} cells per well. Distances between clusters are not quantitative.")
    save(fig, "fig4b_umap", ctx)


# =====================================================================================
# Figure 5 – spatial density per well
# =====================================================================================
def fig_spatial(ctx: Ctx) -> None:
    cx, cy = ctx.cells["Centroid X"].to_numpy(), ctx.cells["Centroid Y"].to_numpy()
    ext = (0.0, float(cx.max()), 0.0, float(cy.max()))
    cm = plt.get_cmap(SEQ_CMAP)
    fig, axes = plt.subplots(3, 12, figsize=(12, 3.9))
    used = {(w.plate_row, w.plate_col): w for w in ctx.wells.itertuples()}
    im = None
    for i, row in enumerate(ROWS):
        for j in range(12):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            w = used.get((row, j + 1))
            if w is None:
                ax.axis("off")
                continue
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(INK if w.qc_status == "high" else "#BBBBBB")
                sp.set_linewidth(1.4 if w.qc_status == "high" else 0.5)
            g = ctx.by_well.get(w.well)
            n = 0 if g is None else len(g)
            ax.set_title(f"{w.well}{flag(w)}  n={_fmt_count(n)}" if n else f"{w.well}", fontsize=6,
                         fontweight="normal", pad=2, loc="center")
            if n == 0:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=NODATA,
                                       hatch="////", edgecolor=HATCH, lw=0))
                ax.text(.5, .5, "no data", transform=ax.transAxes, ha="center", va="center",
                        fontsize=6, color=MUTED)
                continue
            H, _, _ = np.histogram2d(g["Centroid X"], g["Centroid Y"], bins=SPATIAL_BINS,
                                     range=[ext[:2], ext[2:]])
            rel = H / H.mean()
            im = ax.imshow(rel.T, origin="upper", extent=(ext[0], ext[1], ext[3], ext[2]), cmap=cm,
                           vmin=0, vmax=SPATIAL_VMAX, interpolation="nearest")
            ax.set_aspect("equal")
        axes[i, 0].set_ylabel(row, rotation=0, ha="right", va="center", fontsize=8, fontweight="bold")
    for j in range(12):
        axes[0, j].text(.5, 1.32, str(j + 1), transform=axes[0, j].transAxes, ha="center", fontsize=7,
                        fontweight="bold")
    fig.subplots_adjust(wspace=0.08, hspace=0.35, right=0.92)
    if im is not None:
        cax = fig.add_axes([0.935, 0.2, 0.01, 0.6])
        fig.colorbar(im, cax=cax, extend="max",
                     label="Relative local density\n(bin count / mean bin count of the well)")
    u = unit("Centroid X")
    footer(fig, ctx, f"2D histogram of cell centroids ({SPATIAL_BINS} × {SPATIAL_BINS} bins over the common image "
                     f"extent 0–{ext[1]:,.0f} × 0–{ext[3]:,.0f} {u}; image coordinates, y downwards). "
                     "Normalised per well, so colour shows uneven seeding / edge effects independent of total count.")
    save(fig, "fig5_spatial_density", ctx)


# =====================================================================================
# Figure 6 – QC panel (always uses ALL objects, pixel units)
# =====================================================================================
def fig_qc(ctx: Ctx) -> None:
    w = ctx.wells
    p = w[w.present]
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[1.1, 1], hspace=0.45, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    for r in p.itertuples():
        ax.scatter(r.n_total, r.fraction_small * 100, s=34, marker=BLOCK_MARKER[r.block],
                   color=well_color(r.reagent, r.dose), edgecolor=REAGENT_COLOR[r.reagent], lw=1, zorder=3)
        ax.annotate(str(r.well), (r.n_total, r.fraction_small * 100), fontsize=6, color=INK,
                    xytext=(4, 2), textcoords="offset points")
    for thr, lab in [(BORDERLINE_SMALL_FRAC, "borderline"), (HIGH_SMALL_FRAC, "high")]:
        ax.axhline(thr * 100, color=STATUS_COLOR[lab], ls="--", lw=1, zorder=1)
        ax.text(0.99, thr * 100, f"{lab} ≥ {thr:.0%}", transform=blended_transform_factory(ax.transAxes, ax.transData),
                fontsize=6, color=MUTED, va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Segmented objects per well (all, log scale)")
    ax.set_ylabel(f"Objects with Area < {AREA_MIN_PX} px² (%)")
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("a  Object count vs. small-object fraction")
    h = [Line2D([], [], marker="o", lw=0, color=REAGENT_COLOR[r], label=REAGENT_LABEL[r]) for r in REAGENTS]
    h += [Line2D([], [], marker=BLOCK_MARKER[b], lw=0, mfc="white", mec=MUTED, label=BLOCK_SHORT[b]) for b in BLOCKS]
    ax.legend(handles=h, loc="upper left", fontsize=6)

    # b: Area histograms for borderline/high wells vs. pooled ok wells
    ax = fig.add_subplot(gs[0, 1])
    area = ctx.cells_all["Area"].to_numpy(float)
    bins = np.logspace(1, np.log10(max(area.max(), 100)), 70)
    okw = p.well[p.qc_status == "ok"]
    ok_a = ctx.cells_all.loc[ctx.cells_all.well.isin(okw), "Area"].to_numpy(float)
    if len(ok_a):
        hh, _ = np.histogram(ok_a, bins=bins)
        ax.fill_between(bins[:-1], hh / hh.sum() * 100, step="post", color="#D0D0D0", lw=0,
                        label=f"pooled ok wells (n = {len(okw)})")
    flagged = p[p.qc_status != "ok"].sort_values("fraction_small", ascending=False)
    oi = ["#E69F00", "#56B4E9", "#CC79A7", "#009E73", "#0072B2", "#D55E00", "#000000"]
    shown = flagged.head(len(oi))
    for c, r in zip(oi, shown.itertuples()):
        a = ctx.cells_all.loc[ctx.cells_all.well == r.well, "Area"].to_numpy(float)
        hh, _ = np.histogram(a, bins=bins)
        ax.step(bins[:-1], hh / hh.sum() * 100, where="post", color=c, lw=1.3,
                ls="-" if r.qc_status == "high" else "--",
                label=f"well {r.well} ({r.fraction_small:.1%}, {r.qc_status})")
    ax.axvline(AREA_MIN_PX, color=INK, lw=1)
    ax.text(AREA_MIN_PX, 1.0, f" QC cut {AREA_MIN_PX} px²",
            transform=blended_transform_factory(ax.transData, ax.transAxes), fontsize=6, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Area (px², log scale)")
    ax.set_ylabel("Objects per bin (% of well)")
    ax.set_title("b  Area distribution of borderline / high wells")
    ax.legend(loc="upper left", fontsize=6)
    if len(flagged) > len(shown):
        ax.text(1, -0.2, f"{len(flagged) - len(shown)} further flagged wells not shown",
                transform=ax.transAxes, ha="right", fontsize=6, color=MUTED)

    # c: fraction_small for every layout well, value printed on every bar
    ax = fig.add_subplot(gs[1, :])
    for r in w.itertuples():
        if r.present:
            ax.bar(r.well, r.fraction_small * 100, width=0.75, color=STATUS_COLOR[r.qc_status], lw=0)
            ax.text(r.well, r.fraction_small * 100 + 0.4, f"{r.fraction_small * 100:.1f}", ha="center",
                    va="bottom", fontsize=6)
        else:
            ax.bar(r.well, 1.2, width=0.75, color=NODATA, hatch="////", edgecolor=HATCH, lw=0)
            ax.text(r.well, 1.6, "no\ndata", ha="center", va="bottom", fontsize=5, color=MUTED)
    for thr, lab in [(BORDERLINE_SMALL_FRAC, "borderline"), (HIGH_SMALL_FRAC, "high")]:
        ax.axhline(thr * 100, color=STATUS_COLOR[lab], ls="--", lw=1)
    ax.set_xticks(w.well, labels=[f"{r.well}\n{r.plate_row}{r.plate_col}" for r in w.itertuples()], fontsize=6)
    ax.set_xlim(0.4, len(w) + 0.6)
    ax.set_ylim(0, max(p.fraction_small.max() * 100 * 1.15, HIGH_SMALL_FRAC * 100 * 1.2) if len(p) else 20)
    ax.set_xlabel("Well (number / plate position)")
    ax.set_ylabel(f"Objects with Area < {AREA_MIN_PX} px² (%)")
    ax.set_title("c  Small-object fraction per well")
    ax.legend(handles=[Patch(color=STATUS_COLOR["ok"], label=f"ok (< {BORDERLINE_SMALL_FRAC:.0%})"),
                       Patch(color=STATUS_COLOR["borderline"],
                             label=f"borderline ({BORDERLINE_SMALL_FRAC:.0%}–{HIGH_SMALL_FRAC:.0%})"),
                       Patch(color=STATUS_COLOR["high"], label=f"high (≥ {HIGH_SMALL_FRAC:.0%}), {FLAG} in figures"),
                       Patch(facecolor=NODATA, hatch="////", edgecolor=HATCH, label="no data")],
              loc="upper left", ncol=4, fontsize=6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.text(0.01, -0.01, f"QC always uses all segmented objects in pixel units. Wells without data: "
                          f"{fmt_wells(ctx.missing)}.", fontsize=6, color=MUTED, va="top")
    save(fig, "fig6_qc", ctx, suffix=False)


# =====================================================================================
# Figure 7 – composite publication panel
# =====================================================================================
def fig_composite(ctx: Ctx) -> None:
    fig = plt.figure(figsize=(7.4, 10.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.25, 1.25, 1.2, 1.7], hspace=0.62, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    m = draw_plate(ax, ctx.wells, ctx.count_col, _fmt_count, "", vmin=0)
    fig.colorbar(m, ax=ax, fraction=0.02, pad=0.1, label="Cell count")
    ax.set_title("Cell count per well", pad=20)
    panel_letter(ax, "A", x=-0.03, y=1.12)
    axb = [fig.add_subplot(gs[1, 0])]
    axb.append(fig.add_subplot(gs[1, 1], sharey=axb[0]))
    ymax = max(draw_dose_response(a, ctx, b, normalized=True) for a, b in zip(axb, BLOCKS))
    set_dose_ylim(axb, ymax)
    for a, b in zip(axb, BLOCKS):
        a.set_title(BLOCK_SHORT[b], loc="left", pad=14)
    axb[0].set_ylabel("Cell count / auto mean")
    plt.setp(axb[1].get_yticklabels(), visible=False)
    panel_letter(axb[0], "B", y=1.12)
    widths = [len(block_positions(ctx.wells, b)) + 1.4 for b in BLOCKS]
    gsc = gs[2, :].subgridspec(1, 2, width_ratios=widths, wspace=0.05)
    yl = feature_ylim(ctx, "Area")
    axc = [fig.add_subplot(gsc[0, 0])]
    axc.append(fig.add_subplot(gsc[0, 1], sharey=axc[0]))
    for a, b in zip(axc, BLOCKS):
        draw_violins(a, ctx, b, "Area", yl)
        a.set_title(BLOCK_SHORT[b], loc="left", fontsize=8)
    axc[0].set_ylabel(axis_label("Area"))
    plt.setp(axc[1].get_yticklabels(), visible=False)
    panel_letter(axc[0], "C", y=1.1)
    axd = fig.add_subplot(gs[3, 0])
    p = compute_pca(ctx)
    xl, yl2 = _pc_limits(p["sample"])
    _scatter_by(axd, p["sample"], "reagent", REAGENT_COLOR, "Cell density (50% / 90% contours)", xl, yl2)
    axd.set_xlabel(_pc_label(p["pca"], 0))
    axd.set_ylabel(_pc_label(p["pca"], 1))
    panel_letter(axd, "D", y=1.05)
    axd2 = fig.add_subplot(gs[3, 1])
    draw_pca_centroids(axd2, ctx)
    axd2.set_ylabel("")
    axd2.set_title("Per-well median (zoomed)")
    h = [Line2D([], [], color=REAGENT_COLOR[r], marker="o", lw=0, markersize=5, label=REAGENT_LABEL[r])
         for r in REAGENTS]
    h += [Line2D([], [], marker=BLOCK_MARKER[b], lw=0, mfc="white", mec=MUTED, label=BLOCK_SHORT[b]) for b in BLOCKS]
    h.append(Line2D([], [], color=MUTED, ls=":", lw=1, label="auto mean"))
    fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.075))
    footer(fig, ctx, "B: dots = wells, line = mean of replicates; dose in µg / 10⁶ cells, light→dark = 0.5→2 µg. "
                     "C: violin = density, dot = median, bar = IQR.\nD: PCA of 10 z-scored morphology features.",
           y=0.03)
    save(fig, "fig7_composite", ctx)


# =====================================================================================
# main
# =====================================================================================
FIGURES = {"1": fig_plate_heatmap, "2": fig_dose_response, "3": fig_morphology, "4": fig_pca,
           "5": fig_spatial, "6": fig_qc, "7": fig_composite}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-cache", action="store_true", help="ignore the cache and re-read every Excel file")
    ap.add_argument("--include-small", action="store_true",
                    help=f"plot all objects instead of QC-filtered (Area ≥ {AREA_MIN_PX} px²)")
    ap.add_argument("--figures", default=",".join(FIGURES),
                    help="comma-separated figure numbers to make (default: all, e.g. 1,2,6)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    include_small = args.include_small or INCLUDE_SMALL_DEFAULT
    wanted = [f.strip() for f in args.figures.split(",") if f.strip()]
    bad = [f for f in wanted if f not in FIGURES]
    if bad:
        fail(f"unknown figure(s) {bad}; choose from {list(FIGURES)}")

    meta = build_well_meta()
    check_plate_map(meta)
    print("Loading data ...")
    cells, info = load_cells(args.no_cache)
    report_completeness(meta, info)

    cells["is_small"] = cells["Area"] < AREA_MIN_PX     # always in pixel units
    wells = build_well_table(meta, cells)
    sel = cells if include_small else cells[~cells["is_small"]]
    sel = to_physical(sel)
    ctx = Ctx(cells_all=cells, cells=sel, wells=wells, include_small=include_small)
    ctx.by_well = {int(w): g for w, g in sel.groupby("well")}
    print(f"Plotting {ctx.objects_label}: {len(sel):,} of {len(cells):,} objects.")

    export_summary(ctx)
    failed = []
    for key in wanted:
        print(f"Figure {key} ({FIGURES[key].__name__}) ...")
        try:
            FIGURES[key](ctx)
        except Exception:
            traceback.print_exc()
            failed.append(key)
            plt.close("all")
    if failed:
        print(f"\nFigures that failed: {failed}")
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
