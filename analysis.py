#!/usr/bin/env python3
"""
Brightfield (BF/<N>_bf.xlsx) and AF546 fluorescence (Fluo/<N>_af546.xlsx) segmentation
results of the 28-well HIB19 / isotype / IVIG experiment (2026-10-30): caching, QC,
figures 1-7 (BF), 8-14 (fluorescence, combined with BF) and per-well summary tables.

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
from types import SimpleNamespace  # noqa: E402

# =====================================================================================
# CONFIGURATION – edit here
# =====================================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "."            # working folder: plate map (*_patemap.xlsx)
BF_DIR = SCRIPT_DIR / "BF"             # <N>_bf.xlsx      (brightfield segmentation)
FLUO_DIR = SCRIPT_DIR / "Fluo"         # <N>_af546.xlsx   (Alexa Fluor 546 segmentation)
CACHE_DIR = SCRIPT_DIR / "cache"
OUT_DIR = "figures_v2"        # ALL outputs (PNG, SVG, CSV) go here; cache/ stays shared
FIG_DIR = SCRIPT_DIR / OUT_DIR

FILE_PATTERN = re.compile(r"(\d+)_bf\.xlsx$", re.IGNORECASE)
FLUO_FILE_PATTERN = re.compile(r"(\d+)_af546\.xlsx$", re.IGNORECASE)
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

# ---- fluorescence (AF546) -----------------------------------------------------------
# Area density of fluo objects is flat (noise floor) below ~100 px² and rises into the
# cell-size population above it; 100 px² = equivalent diameter 11 px (< 1/3 of the median
# BF cell, 37 px). The BF value (300 px²) is NOT reused: fluo outlines are smaller.
FLUO_AREA_MIN_PX = 100
FLUO_HIGH_SMALL_FRAC = 0.15      # same meaning as HIGH_SMALL_FRAC, for fluo objects (‡)
# Intensity gate ("bright objects"): Mean Intensity >= FLUO_INT_MIN. None = data-driven:
# the FLUO_INT_PERCENTILE of the Mean Intensity of QC-passing fluo objects in the
# reference wells (FLUO_INT_REFERENCE: "auto" = no-antibody wells, "iso" = isotype wells)
# of the SAME block. The other reference is always computed too, as a sensitivity check.
FLUO_INT_MIN = None
FLUO_INT_PERCENTILE = 99.0
FLUO_INT_REFERENCE = "auto"
# BF-to-fluo matching (exploratory)
MATCH_RADIUS_PX = None           # None = 0.5 x median BF equivalent diameter (QC cells)
MATCH_PAIR_SEARCH_PX = 60        # search radius for mutual-NN registration pairs
MATCH_MIN_PAIRS = 50             # registration needs at least this many mutual pairs
MATCH_MAX_AUTO_POS = 0.05        # sanity: auto wells must have <= this BF-positive fraction
MATCH_MIN_BRIGHT_RATE = 0.50     # sanity: median match rate of bright objects (HIB19 wells)
# Intensity spatial analysis: stitched mosaic of 1024 px tiles, 10% overlap
TILE_SIZE_PX = 1024
TILE_OVERLAP = 0.10
TILE_BINS = 16                   # bins per tile axis for the within-tile intensity map
INT_SPATIAL_BINS = 30            # bins per axis for per-well intensity maps
REGION_PX = None                 # region size for the regional intensity model (None = tile step)
REGION_MIN_OBJ = 20              # regions with fewer QC objects get no correction (factor 1)
SPATIAL_TREND_MIN_SHARE = 0.25   # median between-region share of log-intensity variance in the
                                 # auto wells above this = "strong spatial trend" -> show the
                                 # flattened (position-corrected) sensitivity analysis
COMPOSITE_METRIC = "mi_median"   # metric used in fig13 B/C (see FLUO_METRICS)
# Well-edge analysis (fig15). The well is a circle fitted to the object centroids; distance from the
# edge is expressed as a fraction of the fitted radius (0 = edge, 1 = centre).
EDGE_BINS = 20                   # radial bins between 0 and 1
EDGE_ANGLE_BINS = 144            # angular bins used to find the boundary points for the circle fit
EDGE_DIAG_MIN_DEG = 25           # boundary points only from sectors >= this angle away from the x/y axes
                                 # (the imaged square clips the circular well at the axis directions)
EDGE_MIN_OBJ_BIN = 20            # radial bins with fewer fluo objects are not drawn / used
EDGE_GRADIENT_MIN_RHO = 0.30     # median Spearman rho (auto wells) <= -this -> "clear edge gradient"
EDGE_PLATEAU_FRAC = 0.30         # distance >= this fraction = interior plateau (reference for the zone width)
EDGE_TOL = 0.10                  # edge zone ends where the auto profile is within this of the plateau
EDGE_MIN_CENTER_OBJ = 50         # center-only / radial readouts need at least this many fluo objects
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
# Fluorescence tables: the 13 BF columns + two intensity columns (read by NAME).
# The export calls the second one "Integral Intensity"; both spellings are accepted and
# stored as "Integrated Intensity" (= Mean Intensity x Area, verified on load).
MI_COL, II_COL = "Mean Intensity", "Integrated Intensity"
FLUO_REQUIRED_COLS = REQUIRED_COLS + [MI_COL, II_COL]
COLUMN_ALIASES = {II_COL: ["Integrated Intensity", "Integral Intensity"]}
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
    "svg.hashsalt": "analysis",          # deterministic SVG element ids (reproducible files)
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


_EXT = ".parquet" if _parquet_ok() else ".csv"
CACHE_FILE = CACHE_DIR / f"all_cells{_EXT}"          # BF (name kept from the BF-only version)
MANIFEST_FILE = CACHE_DIR / "manifest.json"
FLUO_CACHE_FILE = CACHE_DIR / f"fluo_objects{_EXT}"
FLUO_MANIFEST_FILE = CACHE_DIR / "manifest_fluo.json"


@dataclass
class Modality:
    """Where and how to read one segmentation modality."""
    name: str
    folder: Path
    pattern: re.Pattern
    required: list
    cache_file: Path
    manifest_file: Path


def bf_modality() -> Modality:
    return Modality("BF", BF_DIR, FILE_PATTERN, REQUIRED_COLS, CACHE_FILE, MANIFEST_FILE)


def fluo_modality() -> Modality:
    return Modality("Fluo", FLUO_DIR, FLUO_FILE_PATTERN, FLUO_REQUIRED_COLS, FLUO_CACHE_FILE,
                    FLUO_MANIFEST_FILE)


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


def discover_well_files(mod: Modality | None = None) -> dict[int, Path]:
    mod = mod or bf_modality()
    if not mod.folder.is_dir():
        fail(f"{mod.name} data directory not found: {mod.folder}")
    found: dict[int, Path] = {}
    locks = []
    for p in sorted(mod.folder.iterdir()):
        if p.name.startswith("~$"):          # Excel lock file of an open workbook
            if mod.pattern.search(p.name):
                locks.append(p.name)
            continue
        m = mod.pattern.search(p.name)
        if not m:
            continue
        w = int(m.group(1))
        if not 1 <= w <= N_WELLS_LAYOUT:
            warn(f"{p.name}: well {w} is not in the 1–{N_WELLS_LAYOUT} layout – ignored")
            continue
        if w in found:
            fail(f"two {mod.name} files for well {w}: {found[w].name} and {p.name}")
        found[w] = p
    if locks:
        print(f"  note: ignored Excel lock file(s) in {mod.folder.name}/: {', '.join(locks)} "
              "(workbook open in Excel – the data file itself is still read)")
    if not found:
        fail(f"no files matching '{mod.pattern.pattern}' in {mod.folder}")
    return found


def read_well_file(path: Path, well: int, required: list | None = None) -> pd.DataFrame:
    """Read one per-well table; columns are selected by NAME (never by position)."""
    required = required or REQUIRED_COLS
    try:
        df = pd.read_excel(path, sheet_name=SHEET_NAME, engine=_excel_engine())
    except ValueError as e:
        fail(f"{path.name}: cannot read sheet '{SHEET_NAME}' ({e})")
    for canon, names in COLUMN_ALIASES.items():
        if canon in required and canon not in df.columns:
            alt = [n for n in names if n in df.columns]
            if alt:
                df = df.rename(columns={alt[0]: canon})
    missing = [c for c in required if c not in df.columns]
    if missing:
        fail(f"{path.name}: missing columns {missing} (found {list(df.columns)})")
    df = df[required].copy()
    df.insert(0, "well", np.int16(well))
    return df


def _sig(p: Path) -> dict:
    st = p.stat()
    return {"file": p.name, "mtime_ns": st.st_mtime_ns, "size": st.st_size}


def load_cells(no_cache: bool, mod: Modality | None = None) -> tuple[pd.DataFrame, dict]:
    """Incremental cache: only new/changed files (name, size, mtime) are read."""
    mod = mod or bf_modality()
    files = discover_well_files(mod)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if mod.manifest_file.exists():
        try:
            manifest = json.loads(mod.manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            warn(f"{mod.name} cache manifest unreadable – rebuilding cache")
    prev_run = manifest.get("last_run")

    cached = None
    if not no_cache and mod.cache_file.exists() and manifest.get("files"):
        try:
            cached = _read_table(mod.cache_file)
            if any(c not in cached.columns for c in mod.required):
                warn(f"{mod.name} cache lacks required columns – rebuilding")
                cached = None
        except Exception as e:  # corrupt / partial cache
            warn(f"cache {mod.cache_file.name} unreadable ({e}) – rebuilding")
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

    parts = [cached[cached["well"].isin(keep)][["well"] + mod.required]] if keep else []
    for i, w in enumerate(to_read, 1):
        t0 = time.time()
        df = read_well_file(files[w], w, mod.required)
        parts.append(df)
        print(f"  read {mod.name}/{files[w].name:<18} {len(df):>7,} rows  ({time.time() - t0:.1f} s)  [{i}/{len(to_read)}]")
    cells = pd.concat(parts, ignore_index=True).sort_values(["well", "Cell ID"], ignore_index=True)

    if to_read or removed or cached is None:
        _write_table(cells, mod.cache_file)
    manifest = {
        "files": {str(w): _sig(p) for w, p in files.items()},
        "last_run": {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "wells": sorted(files)},
    }
    mod.manifest_file.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    info = dict(present=sorted(files), prev_run=prev_run, read=to_read, reused=keep,
                changed=changed, removed=removed, full_rebuild=cached is None, label=mod.name)
    return cells, info


def report_completeness(meta: pd.DataFrame, info: dict) -> None:
    present = set(info["present"])
    missing = sorted(set(meta.well) - present)
    print(f"\n=== Data completeness: {info.get('label', 'BF')} " + "=" * 50)
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
    # ---- fluorescence (None when Fluo/ is absent) ----
    fl_all: pd.DataFrame | None = None   # all fluo objects (+ is_small, bright flags)
    fl: pd.DataFrame | None = None       # selected fluo objects (QC or all)
    fw: pd.DataFrame | None = None       # combined per-well table (well_summary_combined)
    fby_well: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    match: dict | None = None
    tile_model: dict | None = None
    edge: dict | None = None            # well-edge model (fits, profiles, zone width)
    edge_cols: pd.DataFrame | None = None  # per-well sensitivity columns (center-only / radial-normalised)

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
        # no creation date in the SVG -> identical files for identical data (reproducibility check)
        for attempt in range(3):     # Windows: a viewer/indexer may hold the file briefly
            try:
                fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300 if ext == "png" else None,
                            metadata={"Date": None} if ext == "svg" else None)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2)
    plt.close(fig)
    print(f"  saved {OUT_DIR}/{stem}.png/.svg")


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
    print(f"  saved {OUT_DIR}/{name}")
    return out


# =====================================================================================
# Figure 1 – plate heatmap
# =====================================================================================
def draw_plate(ax, wells: pd.DataFrame, value_col: str, fmt, title: str, vmin=None,
               flag_fn=None, outline_fn=None, missing_fn=None, norm=None):
    """norm: optional Normalize (e.g. percentile-clipped); flag_fn(row) -> suffix after the well number; outline_fn(row) -> thick outline;
    missing_fn(row) -> text for wells without a value (defaults: BF QC flag / 'no data')."""
    flag_fn = flag_fn or flag
    outline_fn = outline_fn or (lambda r: r.qc_status == "high")
    missing_fn = missing_fn or (lambda r: "no data")
    cm = plt.get_cmap(SEQ_CMAP)
    vals = wells.loc[wells.present, value_col].dropna()
    lo = vals.min() if vmin is None else vmin
    norm = norm or mcolors.Normalize(lo, vals.max() if len(vals) else 1)
    for w in wells.itertuples():
        x, y = w.plate_col - 1, ROWS.index(w.plate_row)
        v = getattr(w, value_col)
        has = bool(w.present) and pd.notna(v)
        fc = cm(norm(v)) if has else NODATA
        ax.add_patch(Rectangle((x + .05, y + .05), .9, .9, facecolor=fc, lw=0,
                               edgecolor=HATCH if not has else "none",
                               hatch=None if has else "////"))
        if outline_fn(w):
            ax.add_patch(Rectangle((x + .05, y + .05), .9, .9, fill=False, edgecolor=INK, lw=1.6))
        tc = text_on(fc)
        ax.text(x + .12, y + .2, f"{w.well}{flag_fn(w)}", fontsize=6, ha="left", va="center", color=tc)
        ax.text(x + .5, y + .54, fmt(v) if has else missing_fn(w), ha="center", va="center",
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
    print(f"  saved {OUT_DIR}/{name}")
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
# FLUORESCENCE (AF546) – checks, thresholds, combined per-well table
#
# Honesty notes (also in the README):
#  * fluo_ratio = n_fluo / n_bf is "fluorescent objects per BF cell", NOT % positive cells:
#    the two counts come from two independent segmentations (ratios > 1 are possible).
#  * Segmentation alone also finds autofluorescent cells (auto wells have many objects),
#    so the intensity gate ("bright" = Mean Intensity >= FLUO_INT_MIN) is essential.
#  * Experimental unit = well (n = 2 replicates); no per-cell p-values anywhere.
# =====================================================================================
FLUO_FLAG = "‡"
FLUO_METRICS = {
    "fluo_ratio": "Fluo objects per BF cell",
    "bright_ratio": "Bright fluo objects per BF cell",
    "mi_median": "Median Mean Intensity (a.u.)",
    "int_per_bf_cell": "Integrated fluorescence per BF cell (a.u.)",
}
# intensity-based replacements for the count-based background logic (used by the *_v2 figures)
FLUO_METRICS_V2 = {
    "mi_median": "Median Mean Intensity (a.u.)",
    "mi_p90": "P90 of Mean Intensity (a.u.)",
    "excess_per_bf_cell": "Excess integrated fluorescence per BF cell (a.u.)",
    "excess_fold": "Excess fluorescence per BF cell, fold over auto",
}
EXTRA_LABELS = {**FLUO_METRICS_V2, "delta_mi_median": "Δ median Mean Intensity vs. auto median (a.u.)"}
METRIC_NEEDS_BF = {"fluo_ratio", "bright_ratio", "int_per_bf_cell", "excess_per_bf_cell", "excess_fold"}


def fluo_sel_label(ctx: Ctx) -> str:
    return "all objects" if ctx.include_small else "QC-filtered"


def metric_label(key: str, ctx: Ctx, kind: str = "raw") -> str:
    lab = {**EXTRA_LABELS, **FLUO_METRICS}[key]
    low = lab[0].lower() + lab[1:]
    if kind == "bgcorr":
        lab = f"Δ {low} vs. auto"
    elif kind == "specific":
        lab = f"HIB19 − iso: {low}"
    return f"{lab}\n({fluo_sel_label(ctx)})"


def flag2(r) -> str:
    """† = BF small-object fraction high, ‡ = fluo small-object fraction high."""
    s = FLAG if getattr(r, "bf_qc_status", "") == "high" else ""
    return s + (FLUO_FLAG if getattr(r, "fluo_qc_status", "") == "high" else "")


def why_missing(r, needs_bf: bool = True) -> str:
    if not r.fluo_present and (not needs_bf or not r.bf_present):
        return "no data" if needs_bf else "no fluo"
    if not r.fluo_present:
        return "no fluo"
    if needs_bf and not r.bf_present:
        return "no BF"
    return "n/a"


def thr_text(ctx: Ctx) -> str:
    t = ctx.thresholds
    parts = [f"{t[b]['primary']:.1f} ({BLOCK_SHORT[b]})" for b in BLOCKS]
    return (f"Bright = Mean Intensity ≥ {' / '.join(parts)} a.u. "
            f"[{t[BLOCKS[0]]['source'].split(' wells')[0]} wells of the same block]")


def fluo_footer(fig, ctx: Ctx, extra: str = "", y: float = -0.01) -> None:
    fw = ctx.fw
    bfsel = "all" if ctx.include_small else f"QC-filtered (Area ≥ {AREA_MIN_PX} px²)"
    flsel = "all" if ctx.include_small else f"QC-filtered (Area ≥ {FLUO_AREA_MIN_PX} px²)"
    lines = [f"BF objects: {bfsel}; fluo objects: {flsel}.  Wells without BF: "
             f"{fmt_wells(fw.well[~fw.bf_present])}; without fluo: {fmt_wells(fw.well[~fw.fluo_present])}.  "
             f"{FLAG} BF / {FLUO_FLAG} fluo small-object fraction ≥ {HIGH_SMALL_FRAC:.0%}.",
             thr_text(ctx) + ".  Intensities in arbitrary units (a.u.); identical exposure assumed for all wells."]
    if extra:
        lines.append(extra)
    fig.text(0.01, y, "\n".join(lines), fontsize=6, color=MUTED, ha="left", va="top")


def fluo_intensity_report(fl: pd.DataFrame) -> dict:
    mi = fl[MI_COL].to_numpy(float)
    ii = fl[II_COL].to_numpy(float)
    a = fl["Area"].to_numpy(float)
    rel = np.abs(ii - mi * a) / np.maximum(np.abs(ii), 1e-12)
    worst = pd.Series(rel).groupby(fl["well"].to_numpy()).max()
    print("\n=== Fluorescence intensity checks ==========================================")
    print(f"Integrated Intensity vs Mean Intensity x Area: max relative deviation {rel.max():.2e} "
          f"(worst well {worst.idxmax()}) -> " + ("holds; Integrated treated as derived (total signal)"
                                                   if rel.max() < 1e-6 else "DOES NOT HOLD"))
    if rel.max() >= 1e-6:
        warn("Integrated Intensity != Mean Intensity x Area – check the export")
    q = np.percentile(mi, [0, 1, 5, 25, 50, 75, 95, 99, 99.9, 100])
    print("Mean Intensity (a.u.) min/P1/P5/P25/median/P75/P95/P99/P99.9/max:\n   "
          + " / ".join(f"{v:.1f}" for v in q))
    mx = mi.max()
    at = mi == mx
    print(f"Maximum {mx:g} is reached by {int(at.sum())} objects (areas {int(a[at].min())}–{int(a[at].max())} px², "
          f"wells {fmt_wells(np.unique(fl['well'].to_numpy()[at]))}); {int((mi >= 0.95 * mx).sum())} objects ≥ 95% of max.")
    for bits in (8, 12, 16):
        print(f"   objects at/above the {bits}-bit maximum ({2 ** bits - 1}): {int((mi >= 2 ** bits - 1).sum())}")
    if at.sum() > 3:
        print(f"   -> a pile-up exactly at {mx:g} (whole objects with constant value) suggests clipping/saturation "
              f"at {mx:g} after intensity scaling; these objects are kept but their intensity is a lower bound.")
    print(f"Objects with Mean Intensity <= 0: {int((mi <= 0).sum())}; minimum {mi.min():.2f} -> "
          + ("values do NOT look background-subtracted." if mi.min() > 0 else "values may be background-subtracted."))
    return dict(max_rel_dev=float(rel.max()), mi_max=float(mx), n_at_max=int(at.sum()))


def compute_thresholds(fl_qc: pd.DataFrame, meta: pd.DataFrame) -> dict:
    res = {}
    for b in BLOCKS:
        r = {}
        for ref in ("auto", "iso"):
            ws = meta.well[(meta.block == b) & (meta.reagent == ref)]
            d = fl_qc[fl_qc.well.isin(ws)]
            r[ref] = float(np.percentile(d[MI_COL], FLUO_INT_PERCENTILE)) if len(d) else np.nan
            r[ref + "_wells"] = sorted(int(x) for x in d.well.unique())
        res[b] = r
    for b in BLOCKS:
        r, ref = res[b], FLUO_INT_REFERENCE
        if FLUO_INT_MIN is not None:
            r["primary"], r["source"] = float(FLUO_INT_MIN), "fixed value (FLUO_INT_MIN)"
        elif np.isfinite(r[ref]):
            r["primary"] = r[ref]
            r["source"] = f"P{FLUO_INT_PERCENTILE:g} of {ref} wells {fmt_wells(r[ref + '_wells'])}"
        else:
            ws = meta.well[meta.reagent == ref]
            d = fl_qc[fl_qc.well.isin(ws)]
            if len(d):
                r["primary"] = float(np.percentile(d[MI_COL], FLUO_INT_PERCENTILE))
                r["source"] = f"P{FLUO_INT_PERCENTILE:g} of {ref} wells of ALL blocks (none in this block)"
                warn(f"no {ref} well with fluo data in block '{b}' – pooled threshold used")
            else:
                r["primary"], r["source"] = np.nan, "unavailable (no reference wells)"
                warn(f"no {ref} wells at all – intensity gate unavailable")
    return res


def build_fluo_table(ctx: Ctx) -> pd.DataFrame:
    bw = ctx.wells.rename(columns={"present": "bf_present", "n_total": "n_bf_total", "n_small": "n_bf_small",
                                   "n_qc": "n_bf_qc", "fraction_small": "bf_fraction_small",
                                   "qc_status": "bf_qc_status"})
    fa = ctx.fl_all
    g = fa.groupby("well")
    t = pd.DataFrame({"n_fluo_total": g.size(), "n_fluo_small": g["is_small"].sum()})
    t["n_fluo_qc"] = t.n_fluo_total - t.n_fluo_small
    t["fluo_fraction_small"] = t.n_fluo_small / t.n_fluo_total
    t["fluo_qc_status"] = np.select([t.fluo_fraction_small >= FLUO_HIGH_SMALL_FRAC,
                                     t.fluo_fraction_small >= BORDERLINE_SMALL_FRAC],
                                    ["high", "borderline"], "ok")
    gs = ctx.fl.groupby("well")
    t["n_fluo_used"] = gs.size()
    t["n_bright"] = gs["bright"].sum()
    t["n_bright_isothr"] = gs["bright_iso"].sum()
    t["frac_bright"] = t.n_bright / t.n_fluo_used
    t["mi_median"] = gs[MI_COL].median()
    t["mi_q1"] = gs[MI_COL].quantile(0.25)
    t["mi_q3"] = gs[MI_COL].quantile(0.75)
    t["ii_median"] = gs[II_COL].median()
    t["ii_total"] = gs[II_COL].sum()
    t["mi_p90"] = gs[MI_COL].quantile(0.90)
    t["excess_ii_total"] = gs["excess_ii"].sum()
    # sensitivity analysis (spatially flattened intensities) – clearly named, never replaces raw
    if "mi_flat" in ctx.fl.columns:
        t["mi_median_flat_sensitivity"] = gs["mi_flat"].median()
        t["n_bright_flat_sensitivity"] = gs["bright_flat"].sum()
        t["frac_bright_flat_sensitivity"] = t.n_bright_flat_sensitivity / t.n_fluo_used
    if ctx.tile_model is not None:
        t["spatial_between_region_share"] = pd.Series(ctx.tile_model["share"])
    w = bw.merge(t, left_on="well", right_index=True, how="left")
    w["fluo_present"] = w.n_fluo_total.notna()
    w["bf_qc_status"] = w.bf_qc_status.where(w.bf_present, "no data")
    w["fluo_qc_status"] = w.fluo_qc_status.fillna("no data")
    nbf = (w.n_bf_total if ctx.include_small else w.n_bf_qc).where(lambda s: s > 0)
    w["n_bf_used"] = nbf
    w["fluo_ratio"] = w.n_fluo_used / nbf
    w["fluo_ratio_gt1"] = w.fluo_ratio > 1
    w["bright_ratio"] = w.n_bright / nbf
    w["bright_ratio_isothr"] = w.n_bright_isothr / nbf
    w["int_per_bf_cell"] = w.ii_total / nbf
    w["excess_per_bf_cell"] = w.excess_ii_total / nbf
    # area normalisation: BF-segmented area (QC BF objects) and fluo-segmented area (QC fluo objects), px²
    w["bf_area_qc"] = w.well.map(ctx.cells_all[~ctx.cells_all.is_small].groupby("well")["Area"].sum())
    w["fluo_area_qc"] = w.well.map(ctx.fl_all[~ctx.fl_all.is_small].groupby("well")["Area"].sum())
    w.loc[~w.bf_present.astype(bool), "bf_area_qc"] = np.nan
    w["ii_per_bf_area"] = w.ii_total / w.bf_area_qc.where(w.bf_area_qc > 0)
    if ctx.edge_cols is not None:
        w = w.merge(ctx.edge_cols, on="well", how="left")
    w["mi_auto_ref"] = w.block.map({b: ctx.thresholds[b]["mi_auto_ref"] for b in BLOCKS})
    w["delta_mi_median"] = w.mi_median - w.mi_auto_ref
    _ab = w[w.reagent == "auto"].groupby("block")["excess_per_bf_cell"].mean()
    w["excess_auto_mean"] = w.block.map(_ab)
    w["excess_fold"] = w.excess_per_bf_cell / w.excess_auto_mean.where(w.excess_auto_mean > 0)
    if "n_bright_flat_sensitivity" in w.columns:
        w["bright_ratio_flat_sensitivity"] = w.n_bright_flat_sensitivity / nbf
    th = ctx.thresholds
    w["int_threshold"] = w.block.map({b: th[b]["primary"] for b in BLOCKS})
    w["int_threshold_auto"] = w.block.map({b: th[b]["auto"] for b in BLOCKS})
    w["int_threshold_iso"] = w.block.map({b: th[b]["iso"] for b in BLOCKS})
    w["int_threshold_source"] = w.block.map({b: th[b]["source"] for b in BLOCKS})
    # background correction (auto wells of the same block) and specific signal (HIB19 - iso)
    for m in list(FLUO_METRICS) + ["frac_bright", "mi_p90", "delta_mi_median", "excess_per_bf_cell", "excess_fold",
              "ii_total", "ii_per_bf_area"]:
        base = w[w.reagent == "auto"].groupby("block")[m].agg(["mean", "count"])
        w[f"{m}_auto_mean"] = w.block.map(base["mean"])
        w[f"{m}_auto_n"] = w.block.map(base["count"]).fillna(0).astype(int)
        w[f"{m}_bgcorr"] = w[m] - w[f"{m}_auto_mean"]
        if m in ("ii_total", "ii_per_bf_area", "mi_median"):
            w[f"{m}_fold"] = w[m] / w[f"{m}_auto_mean"].where(w[f"{m}_auto_mean"] > 0)
        spec, ref = [], []
        for r in w.itertuples():
            val = getattr(r, m)
            if r.reagent != "HIB19" or pd.isna(val):
                spec.append(np.nan)
                ref.append("")
                continue
            partner = r.well + 6        # isotype well of the same block, dose and replicate
            pv = w.loc[w.well == partner, m]
            if len(pv) and pd.notna(pv.iloc[0]):
                spec.append(val - pv.iloc[0])
                ref.append(f"iso well {partner}")
            else:
                iso = w[(w.block == r.block) & (w.reagent == "iso") & (w.dose == r.dose) & w[m].notna()]
                if len(iso):
                    spec.append(val - iso[m].mean())
                    ref.append(f"iso mean (wells {fmt_wells(iso.well)})")
                else:
                    spec.append(np.nan)
                    ref.append("no iso value at this block/dose")
        w[f"{m}_specific"] = spec
        w[f"{m}_specific_ref"] = ref
    w["objects_used"] = "all" if ctx.include_small else "qc_filtered"
    return w


def _region_key(df: pd.DataFrame, size: float, ox: float = 0.0, oy: float = 0.0) -> np.ndarray:
    return (((df["Centroid X"].to_numpy(float) - ox) // size).astype(np.int64) * 100_000
            + ((df["Centroid Y"].to_numpy(float) - oy) // size).astype(np.int64))


def _between_share(d: pd.DataFrame, size: float, ox: float = 0.0, oy: float = 0.0) -> float:
    """Share of the variance of log(Mean Intensity) explained by region medians."""
    lm = pd.Series(np.log(d[MI_COL].to_numpy(float)))
    key = _region_key(d, size, ox, oy)
    g = lm.groupby(key)
    med, n = g.transform("median"), g.transform("size")
    ok = (n >= REGION_MIN_OBJ).to_numpy()
    if ok.sum() < 50 or lm[ok].var() == 0:
        return np.nan
    return float(1 - (lm[ok] - med[ok]).var() / lm[ok].var())


def compute_tile_model(ctx: Ctx) -> dict:
    """Spatial intensity model (QC-filtered fluo objects, pixel coordinates).

    1. Within-tile pattern (vignetting / seams): Mean Intensity / well median as a function of
       the position inside a tile (step = TILE_SIZE_PX x (1 - TILE_OVERLAP)). The tile origin is
       unknown, which only shifts the pattern. Objects lie independently of the tile grid, so
       biology averages out and any pattern is an imaging artefact.
    2. Tile-offset test: share of log-intensity variance between grid cells for a grid aligned
       to the assumed tiles vs. the same grid shifted by half a tile. Tile-wise offsets would
       make the aligned grid clearly better.
    3. Regional trend: per well, median Mean Intensity in REGION_PX squares. The flattened
       intensity Mean Intensity / (region median / well median) removes spatial variation
       WITHIN a well but keeps its overall level. Used ONLY as a sensitivity analysis.
       Caveat: if truly positive cells cluster spatially, flattening also removes real signal.
    """
    fl = ctx.fl_all[~ctx.fl_all.is_small]
    step = TILE_SIZE_PX * (1 - TILE_OVERLAP)
    region = float(REGION_PX) if REGION_PX else step
    med = fl.groupby("well")[MI_COL].transform("median")
    rel = (fl[MI_COL] / med).to_numpy(float)
    u = (fl["Centroid X"].to_numpy(float) % step) / step
    v = (fl["Centroid Y"].to_numpy(float) % step) / step
    iu = np.minimum((u * TILE_BINS).astype(int), TILE_BINS - 1)
    iv = np.minimum((v * TILE_BINS).astype(int), TILE_BINS - 1)
    df = pd.DataFrame({"iu": iu, "iv": iv, "rel": rel})
    agg = df.groupby(["iv", "iu"]).rel.agg(["median", "size"])
    M = np.full((TILE_BINS, TILE_BINS), np.nan)
    ok = agg["size"] >= 20
    for (a, b), val in agg.loc[ok, "median"].items():
        M[a, b] = val
    center = np.nanmedian(M)
    amp = (np.nanpercentile(M, 95) - np.nanpercentile(M, 5)) / center
    prof_u = df.groupby("iu").rel.median().reindex(range(TILE_BINS)).to_numpy()
    prof_v = df.groupby("iv").rel.median().reindex(range(TILE_BINS)).to_numpy()
    # 2 + 3: per-well between-region share (aligned vs half-tile shifted grid) and region medians
    share, phase, factors = {}, [], {}
    for w, d in fl.groupby("well"):
        share[int(w)] = _between_share(d, region)
        phase.append(dict(well=int(w), aligned=_between_share(d, step),
                          shifted=np.nanmean([_between_share(d, step, step / 2, 0),
                                              _between_share(d, step, 0, step / 2),
                                              _between_share(d, step, step / 2, step / 2)])))
        key = _region_key(d, region)
        s = pd.Series(d[MI_COL].to_numpy(float)).groupby(key).agg(["median", "size"])
        f = (s["median"] / d[MI_COL].median()).where(s["size"] >= REGION_MIN_OBJ)
        factors[int(w)] = f.dropna().to_dict()
    auto_wells = ctx.wells.well[ctx.wells.reagent == "auto"]
    auto_share = [share[w] for w in auto_wells if w in share and np.isfinite(share[w])]
    med_share = float(np.median(auto_share)) if auto_share else np.nan
    return dict(map=M, amp=float(amp), prof_u=prof_u, prof_v=prof_v, step=step, region=region, n=int(len(fl)),
                share=share, phase=pd.DataFrame(phase), factors=factors, auto_share=med_share,
                strong=bool(np.isfinite(med_share) and med_share >= SPATIAL_TREND_MIN_SHARE))


def apply_flattening(df: pd.DataFrame, model: dict) -> pd.Series:
    """Spatially flattened Mean Intensity (sensitivity analysis only – raw values are kept)."""
    out = df[MI_COL].to_numpy(float).copy()
    key = _region_key(df, model["region"])
    wells = df["well"].to_numpy()
    for w in np.unique(wells):
        m = wells == w
        f = pd.Series(key[m]).map(model["factors"].get(int(w), {})).fillna(1.0).to_numpy(float)
        out[m] = out[m] / f
    return pd.Series(out, index=df.index)


def _kasa(px, py):
    A = np.c_[2 * px, 2 * py, np.ones(len(px))]
    sol = np.linalg.lstsq(A, px ** 2 + py ** 2, rcond=None)[0]
    return float(sol[0]), float(sol[1]), float(np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2))


def fit_well_circle(x: np.ndarray, y: np.ndarray) -> dict:
    """Circle (centre, radius) of the well from object centroids (pixels).

    The imaged mosaic is a square that is smaller than the circular well, so it clips the well at the
    x/y axis directions; only the four diagonal sectors show the true well boundary. Method: in
    EDGE_ANGLE_BINS angular bins inside the diagonal sectors take the most distant centroid (= nearest
    to the wall), fit a circle to these boundary points (algebraic Kasa fit), and iterate with the
    centre re-estimated and boundary points beyond 2.5 robust SD (MAD) of the residuals dropped.
    The fit residual is the RMS radial deviation of the retained boundary points."""
    nb = EDGE_ANGLE_BINS
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    for _ in range(6):
        ang = np.arctan2(y - cy, x - cx)
        r = np.hypot(x - cx, y - cy)
        a = np.abs(((ang + np.pi / 4) % (np.pi / 2)) - np.pi / 4)     # 0 on the axes, 45° on the diagonals
        idx = np.flatnonzero(a > np.radians(EDGE_DIAG_MIN_DEG))
        k = np.floor((ang[idx] + np.pi) / (2 * np.pi) * nb).astype(int).clip(0, nb - 1)
        df = pd.DataFrame({"k": k, "r": r[idx], "i": idx})
        sel = df.loc[df.groupby("k").r.idxmax(), "i"].to_numpy()
        px, py = x[sel], y[sel]
        keep = np.ones(len(px), bool)
        for _ in range(4):
            cx2, cy2, R = _kasa(px[keep], py[keep])
            res = np.hypot(px - cx2, py - cy2) - R
            med = np.median(res[keep])
            mad = 1.4826 * np.median(np.abs(res[keep] - med))
            keep = np.abs(res - med) <= max(2.5 * mad, 50.0)
        cx, cy = cx2, cy2
    cx, cy, R = _kasa(px[keep], py[keep])
    res = np.hypot(px[keep] - cx, py[keep] - cy) - R
    return dict(cx=cx, cy=cy, R=R, rms=float(np.sqrt((res ** 2).mean())), n_boundary=int(keep.sum()), n_boundary_all=len(px))


def _rank_corr(a, b) -> float:
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    return float(a.rank().corr(b.rank())) if len(a) > 3 else np.nan


def _edge_frac(df: pd.DataFrame, fits: dict) -> np.ndarray:
    cx = df.well.map({w: f["cx"] for w, f in fits.items()}).to_numpy(float)
    cy = df.well.map({w: f["cy"] for w, f in fits.items()}).to_numpy(float)
    R = df.well.map({w: f["R"] for w, f in fits.items()}).to_numpy(float)
    return 1.0 - np.hypot(df["Centroid X"].to_numpy(float) - cx, df["Centroid Y"].to_numpy(float) - cy) / R


def _bin_of(frac: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(frac * EDGE_BINS), 0, EDGE_BINS - 1).astype(int)       # outside the circle -> bin 0


def compute_edge_model(ctx: Ctx, fl: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Fit the well circle per well, assign distance-from-edge to every BF and fluo object, build radial
    profiles and the per-well edge summary. Adds `edge_frac` to ctx.cells_all and `fl`."""
    bf = ctx.cells_all
    fits = {}
    bfg = {int(w): g[["Centroid X", "Centroid Y"]] for w, g in bf.groupby("well")}
    flg = {int(w): g[["Centroid X", "Centroid Y"]] for w, g in fl.groupby("well")}
    for w in sorted(set(bfg) | set(flg)):
        xy = pd.concat([bfg.get(w), flg.get(w)]).to_numpy(float)
        if len(xy) >= 500:
            fits[w] = fit_well_circle(xy[:, 0], xy[:, 1])
    bf["edge_frac"] = _edge_frac(bf, fits)
    fl["edge_frac"] = _edge_frac(fl, fits)
    # imaged area per radial bin (the square mosaic clips the circle): grid of points over the mosaic extent
    xm = max(bf["Centroid X"].max(), fl["Centroid X"].max())
    ym = max(bf["Centroid Y"].max(), fl["Centroid Y"].max())
    step = 100.0
    gx, gy = np.meshgrid(np.arange(step / 2, xm, step), np.arange(step / 2, ym, step))
    edges = np.linspace(0, 1, EDGE_BINS + 1)
    mset = meta.set_index("well")
    bfq = bf[~bf.is_small]
    flq = fl[~fl.is_small]
    prof, summ = [], []
    for w, f in fits.items():
        fr = 1 - np.hypot(gx - f["cx"], gy - f["cy"]) / f["R"]
        imaged = np.histogram(fr[(fr >= 0) & (fr < 1)], bins=edges)[0] * step ** 2
        b = bfq[bfq.well == w]
        o = flq[flq.well == w]
        kb, ko = _bin_of(b["edge_frac"].to_numpy(float)), _bin_of(o["edge_frac"].to_numpy(float))
        area = pd.Series(b["Area"].to_numpy(float)).groupby(kb).sum().reindex(range(EDGE_BINS)).fillna(0).to_numpy()
        nb_ = np.bincount(kb, minlength=EDGE_BINS)
        mi = pd.Series(o[MI_COL].to_numpy(float)).groupby(ko)
        med, q1, q3 = mi.median(), mi.quantile(0.25), mi.quantile(0.75)
        nf = np.bincount(ko, minlength=EDGE_BINS)
        for i in range(EDGE_BINS):
            prof.append(dict(well=w, bin=i, d_lo=edges[i], d_hi=edges[i + 1], imaged_area_px2=imaged[i],
                             n_bf=int(nb_[i]), bf_area_px2=area[i],
                             bf_coverage=area[i] / imaged[i] if imaged[i] > 0 else np.nan,
                             bf_density_per_1e6px2=nb_[i] / imaged[i] * 1e6 if imaged[i] > 0 else np.nan,
                             n_fluo=int(nf[i]), mi_median=med.get(i, np.nan), mi_q1=q1.get(i, np.nan),
                             mi_q3=q3.get(i, np.nan)))
        ok = nf >= EDGE_MIN_OBJ_BIN
        pw = pd.DataFrame(prof[-EDGE_BINS:])
        edge_m = o.loc[o.edge_frac < 0.10, MI_COL].median()
        plat_m = o.loc[o.edge_frac >= EDGE_PLATEAU_FRAC, MI_COL]
        zone = pw[pw.d_hi <= 0.10 + 1e-9]
        plat = pw[pw.d_lo >= EDGE_PLATEAU_FRAC]
        summ.append(dict(
            well=w, reagent=mset.reagent.get(w), dose=mset.dose.get(w), block=mset.block.get(w),
            replicate=mset.replicate.get(w), cx_px=f["cx"], cy_px=f["cy"], radius_px=f["R"],
            fit_rms_px=f["rms"], fit_rms_frac_of_R=f["rms"] / f["R"], n_boundary_points=f["n_boundary"],
            n_fluo_qc=len(o), frac_fluo_outside_circle=float((o.edge_frac < 0).mean()) if len(o) else np.nan,
            spearman_MI_vs_edge_distance=_rank_corr(o[MI_COL], o["edge_frac"]) if len(o) > 3 else np.nan,
            spearman_binmedian_vs_distance=_rank_corr(pw.mi_median[ok], pw.d_lo[ok] + 0.025) if ok.sum() > 3 else np.nan,
            mi_median_edge_zone_lt_0p10=edge_m, mi_median_interior_ge_0p30=plat_m.median() if len(plat_m) >= 50 else np.nan,
            n_fluo_interior_ge_0p30=len(plat_m),
            bf_coverage_edge_zone_lt_0p10=zone.bf_area_px2.sum() / zone.imaged_area_px2.sum(),
            bf_coverage_interior_ge_0p30=plat.bf_area_px2.sum() / plat.imaged_area_px2.sum()))
    prof = pd.DataFrame(prof)
    summ = pd.DataFrame(summ)
    Rmed = summ.radius_px.median()
    summ["radius_rel_to_median"] = summ.radius_px / Rmed - 1
    summ["fit_flag"] = np.where((summ.fit_rms_frac_of_R > 0.03) | (summ.radius_rel_to_median.abs() > 0.03), "check", "ok")
    summ["edge_to_interior_MI_ratio"] = summ.mi_median_edge_zone_lt_0p10 / summ.mi_median_interior_ge_0p30
    auto = summ[summ.reagent == "auto"]
    rho_med = float(auto.spearman_MI_vs_edge_distance.median())
    ratio_med = float(auto.edge_to_interior_MI_ratio.median())
    clear = bool(rho_med <= -EDGE_GRADIENT_MIN_RHO and ratio_med >= 1 + EDGE_TOL)
    # zone width X: first radial bin where the pooled, plateau-normalised auto profile is within EDGE_TOL
    norm_prof = []
    for r in auto.itertuples():
        pw = prof[prof.well == r.well]
        if np.isfinite(r.mi_median_interior_ge_0p30):
            v = pw.mi_median.where(pw.n_fluo >= EDGE_MIN_OBJ_BIN) / r.mi_median_interior_ge_0p30
            norm_prof.append(v.to_numpy())
    pooled = np.nanmedian(np.vstack(norm_prof), axis=0) if norm_prof else np.full(EDGE_BINS, np.nan)
    X = EDGE_PLATEAU_FRAC
    for i in range(EDGE_BINS):
        if np.isfinite(pooled[i]) and pooled[i] <= 1 + EDGE_TOL:
            X = i / EDGE_BINS
            break
    ctx.edge = dict(fits=fits, profiles=prof, summary=summ, X=X, clear=clear, rho_med=rho_med, ratio_med=ratio_med,
                    pooled_auto_profile=pooled, rmed=Rmed, mosaic=(xm, ym))
    print("\n=== Well-edge analysis ======================================================")
    print(f"Circle fit (boundary points of the diagonal sectors, {EDGE_ANGLE_BINS} angular bins, robust Kasa fit): "
          f"radius median {Rmed:.0f} px (range {summ.radius_px.min():.0f}–{summ.radius_px.max():.0f}); "
          f"fit RMS {summ.fit_rms_px.median():.0f} px median, max {summ.fit_rms_px.max():.0f} px "
          f"({summ.fit_rms_frac_of_R.max():.1%} of R); flagged 'check': {fmt_wells(summ.well[summ.fit_flag == 'check'])}")
    print(f"Auto wells: median Spearman rho (Mean Intensity vs distance from edge) = {rho_med:.2f}; "
          f"edge-zone (<0.10 R) / interior (>=0.30 R) median-MI ratio = {ratio_med:.2f} -> "
          + ("CLEAR edge gradient" if clear else "no clear edge gradient"))
    print(f"Edge-zone width from the pooled auto profile (within {EDGE_TOL:.0%} of the interior plateau): X = {X:.2f} R")
    print("=============================================================================\n")


def edge_sensitivity(ctx: Ctx, sel: pd.DataFrame, meta: pd.DataFrame) -> None:
    """SENSITIVITY ANALYSIS (raw values stay primary): per-well readouts from objects in the central zone only
    (distance from edge >= X) and Mean Intensity normalised per radial bin by the median of the auto wells of the
    same block. Stored in ctx.edge_cols (merged into well_summary_combined.csv)."""
    E = ctx.edge
    if E is None or not E["clear"]:
        ctx.edge_cols = None
        return
    X = E["X"]
    mset = meta.set_index("well").block
    bfq = ctx.cells_all[~ctx.cells_all.is_small]
    bfq = bfq[bfq.edge_frac >= X]
    bf_center_area = bfq.groupby("well")["Area"].sum()
    # radial reference: median MI of QC auto-well objects per (block, radial bin)
    qa = ctx.fl_all[~ctx.fl_all.is_small]
    qa = qa[qa.well.map(meta.set_index("well").reagent) == "auto"]
    ref = {}
    for (b, k), g in qa.assign(block=qa.well.map(mset), k=_bin_of(qa.edge_frac.to_numpy(float))).groupby(["block", "k"]):
        if len(g) >= EDGE_MIN_OBJ_BIN:
            ref[(b, k)] = g[MI_COL].median()
    sel = sel.assign(block=sel.well.map(mset))
    sel["ref"] = [ref.get((b, k), np.nan) for b, k in zip(sel.block, _bin_of(sel.edge_frac.to_numpy(float)))]
    sel["mi_rel"] = sel[MI_COL] / sel["ref"]
    rows = []
    for w, g in sel.groupby("well"):
        c = g[g.edge_frac >= X]
        r = g.dropna(subset=["mi_rel"])
        ok_c = len(c) >= EDGE_MIN_CENTER_OBJ
        rows.append(dict(
            well=int(w), n_fluo_center_only=len(c),
            mi_median_center_only=c[MI_COL].median() if ok_c else np.nan,
            mi_p90_center_only=c[MI_COL].quantile(0.9) if ok_c else np.nan,
            bf_area_center_only=float(bf_center_area.get(w, np.nan)),
            ii_per_bf_area_center_only=(c[II_COL].sum() / bf_center_area.get(w, np.nan)) if ok_c and bf_center_area.get(w, 0) > 0 else np.nan,
            n_fluo_radial_norm=len(r),
            mi_median_radial_norm=r.mi_rel.median() if len(r) >= EDGE_MIN_CENTER_OBJ else np.nan,
            mi_p90_radial_norm=r.mi_rel.quantile(0.9) if len(r) >= EDGE_MIN_CENTER_OBJ else np.nan))
    ctx.edge_cols = pd.DataFrame(rows)


def prepare_fluo(ctx: Ctx, meta: pd.DataFrame, fl: pd.DataFrame) -> None:
    fluo_intensity_report(fl)
    fl = fl.copy()
    fl["is_small"] = fl["Area"] < FLUO_AREA_MIN_PX              # always in pixel units
    ctx.thresholds = compute_thresholds(fl[~fl.is_small], meta)
    blk = fl.well.map(meta.set_index("well").block)
    thr = blk.map({b: ctx.thresholds[b]["primary"] for b in BLOCKS}).to_numpy(float)
    thr_iso = blk.map({b: ctx.thresholds[b]["iso"] for b in BLOCKS}).to_numpy(float)
    mi = fl[MI_COL].to_numpy(float)
    # autofluorescence reference: median Mean Intensity of all QC fluo objects in the auto wells of the block
    qc_all = fl[~fl.is_small]
    for b in BLOCKS:
        ws = meta.well[(meta.block == b) & (meta.reagent == "auto")]
        v = qc_all.loc[qc_all.well.isin(ws), MI_COL]
        ctx.thresholds[b]["mi_auto_ref"] = float(v.median()) if len(v) else np.nan
    ref = blk.map({b: ctx.thresholds[b]["mi_auto_ref"] for b in BLOCKS}).to_numpy(float)
    # excess integrated intensity per object = max(MeanIntensity - MI_auto_median_of_block, 0) * Area
    fl["excess_ii"] = np.where(np.isfinite(ref), np.maximum(mi - ref, 0.0) * fl["Area"].to_numpy(float), np.nan)
    fl["bright"] = np.where(np.isfinite(thr), mi >= thr, False)
    fl["bright_iso"] = np.where(np.isfinite(thr_iso), mi >= thr_iso, False)
    ctx.fl_all = fl
    ctx.tile_model = compute_tile_model(ctx)
    # sensitivity analysis: flattened intensity + its own P99-of-auto gate (extra columns, raw untouched)
    fl["mi_flat"] = apply_flattening(fl, ctx.tile_model)
    qf = fl[~fl.is_small]
    thr_flat = {}
    for b in BLOCKS:
        ws = meta.well[(meta.block == b) & (meta.reagent == FLUO_INT_REFERENCE)]
        v = qf.loc[qf.well.isin(ws), "mi_flat"]
        thr_flat[b] = float(np.percentile(v, FLUO_INT_PERCENTILE)) if len(v) else ctx.thresholds[b]["primary"]
        ctx.thresholds[b]["flat"] = thr_flat[b]
    tf = blk.map(thr_flat).to_numpy(float)
    fl["bright_flat"] = np.where(np.isfinite(tf), fl["mi_flat"].to_numpy(float) >= tf, False)
    compute_edge_model(ctx, fl, meta)
    sel = fl if ctx.include_small else fl[~fl.is_small]
    edge_sensitivity(ctx, sel, meta)
    ctx.fl = to_physical(sel)
    ctx.fby_well = {int(w): g for w, g in ctx.fl.groupby("well")}
    ctx.fw = build_fluo_table(ctx)

    print("\n=== Fluorescence intensity gate =============================================")
    for b in BLOCKS:
        t = ctx.thresholds[b]
        print(f"{b:<24} FLUO_INT_MIN = {t['primary']:.2f} a.u.  [{t['source']}]   "
              f"alternative P{FLUO_INT_PERCENTILE:g}: auto {t['auto']:.2f}, iso {t['iso']:.2f}")
    tm = ctx.tile_model
    ph = tm["phase"]
    print(f"Within-tile intensity pattern (step {tm['step']:.1f} px): amplitude (P95−P5)/median = {tm['amp']:.3f}")
    print(f"Tile-offset test: between-cell share of log-intensity variance, tile-aligned grid "
          f"{ph.aligned.median():.1%} vs half-tile-shifted grid {ph.shifted.median():.1%} (median over wells) -> "
          + ("tile-aligned offsets likely" if ph.aligned.median() - ph.shifted.median() > 0.05
             else "no evidence for tile-aligned offsets"))
    print(f"Regional trend ({tm['region']:.0f} px regions): median between-region share in auto wells "
          f"{tm['auto_share']:.1%} -> {'STRONG' if tm['strong'] else 'weak'} (threshold {SPATIAL_TREND_MIN_SHARE:.0%}); "
          f"flattened-intensity gate (sensitivity only): "
          + " / ".join(f"{ctx.thresholds[b]['flat']:.2f} ({BLOCK_SHORT[b]})" for b in BLOCKS))
    fw = ctx.fw
    both = fw.well[fw.bf_present & fw.fluo_present]
    print(f"\nWells with both modalities: {fmt_wells(both)}; BF only: {fmt_wells(fw.well[fw.bf_present & ~fw.fluo_present])}; "
          f"fluo only: {fmt_wells(fw.well[~fw.bf_present & fw.fluo_present])}; neither: "
          f"{fmt_wells(fw.well[~fw.bf_present & ~fw.fluo_present])}")
    print(f"\n{'well':>4} {'reagent':<6} {'dose':>4} {'block':<10} {'n_bf':>7} {'n_fluo':>7} {'n_bright':>8} "
          f"{'fl/bf':>6} {'bright/bf':>9} {'medMI':>6}  (objects: {fluo_sel_label(ctx)})")
    for r in fw.itertuples():
        def f(v, fmt):
            width = int(re.match(r"\d+", fmt).group())
            return format(v, fmt) if pd.notna(v) else "–".rjust(width)
        print(f"{r.well:>4} {r.reagent:<6} {r.dose:>4g} {BLOCK_SHORT[r.block]:<10} {f(r.n_bf_used, '7.0f')} "
              f"{f(r.n_fluo_used, '7.0f')} {f(r.n_bright, '8.0f')} {f(r.fluo_ratio, '6.3f')} "
              f"{f(r.bright_ratio, '9.4f')} {f(r.mi_median, '6.2f')}{'  ratio>1' if r.fluo_ratio_gt1 else ''}")
    print("=============================================================================\n")


def export_fluo_summary(ctx: Ctx) -> None:
    name = "well_summary_combined" + ("_allobjects" if ctx.include_small else "") + ".csv"
    ctx.fw.to_csv(FIG_DIR / name, index=False, float_format="%.6g")
    print(f"  saved {OUT_DIR}/{name}")
    if ctx.edge is not None:
        en = "edge_effect_summary" + ("_allobjects" if ctx.include_small else "") + ".csv"
        ctx.edge["summary"].to_csv(FIG_DIR / en, index=False, float_format="%.6g")
        pn = "edge_effect_radial_profiles" + ("_allobjects" if ctx.include_small else "") + ".csv"
        ctx.edge["profiles"].to_csv(FIG_DIR / pn, index=False, float_format="%.6g")
        print(f"  saved {OUT_DIR}/{en}, {OUT_DIR}/{pn}")


# ---- shared plotting helpers --------------------------------------------------------
def _fmt_ratio(v):
    return f"{v:.2f}" + ("*" if v > 1 else "")


def _fmt_small(v):
    if v >= 0.1:
        return f"{v:.2f}"
    return f"{v:.3f}" if v >= 0.001 or v == 0 else f"{v:.4f}"


def _shared_ylim(axes, ranges, zero_floor: bool) -> None:
    lo = np.nanmin([r[0] for r in ranges] + [np.nan]) if ranges else np.nan
    hi = np.nanmax([r[1] for r in ranges] + [np.nan]) if ranges else np.nan
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return
    if zero_floor and lo >= 0:
        lo = 0.0
    if not zero_floor:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    pad = (hi - lo) * 0.10 or abs(hi) * 0.1 or 1.0
    for ax in axes:
        ax.set_ylim(lo - (0 if (zero_floor and lo == 0) else pad), hi + pad)


def draw_metric_dose(ax, ctx: Ctx, block: str, col: str, metric: str, ref: str = "auto",
                     label_wells: bool = True):
    """Generic per-well dose plot (dots = wells, line = mean of replicates)."""
    fw = ctx.fw
    lay = fw[fw.block == block]
    sub = lay[lay[col].notna()]
    needs_bf = metric in METRIC_NEEDS_BF
    if ref == "auto":
        a = sub[sub.reagent == "auto"]
        if len(a):
            ax.axhline(a[col].mean(), color=MUTED, ls=":", lw=1, zorder=1)
        ax.text(0.0, 1.01, f"auto: n = {len(a)} well{'s' if len(a) != 1 else ''}"
                + (f" ({fmt_wells(a.well)})" if len(a) else ""), transform=ax.transAxes,
                fontsize=5.5, color=MUTED, va="bottom")
    else:
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    for reagent in REAGENTS:
        r = sub[sub.reagent == reagent]
        if r.empty:
            continue
        c = REAGENT_COLOR[reagent]
        xs = np.array([_xpos(x) for x in r.dose]) + R_OFFSET[reagent] + (r.replicate - 1.5) * 0.07
        ax.scatter(xs, r[col], s=24, color=[well_color(reagent, x) for x in r.dose], edgecolor=c,
                   lw=0.9, zorder=3, marker=BLOCK_MARKER[block])
        if reagent != "auto":
            means = r.groupby("dose")[col].mean().reindex(DOSES)
            mx = np.array([_xpos(x) for x in DOSES]) + R_OFFSET[reagent]
            ax.plot(mx, means.values, color=c, lw=1.8, zorder=2, solid_capstyle="round")
            ax.scatter(mx, means.values, marker="_", s=90, color=c, lw=1.8, zorder=2)
        if label_wells:
            for xx, yy, row in zip(xs, r[col], r.itertuples()):
                left = reagent == "HIB19" or (reagent == "auto" and row.replicate == 1)
                ax.annotate(f"{row.well}{flag2(row)}", (xx, yy), fontsize=5, color=MUTED,
                            xytext=(-4 if left else 4, 0), textcoords="offset points",
                            ha="right" if left else "left", va="center")
    notes = []
    reagents = ["HIB19"] if col.endswith("_specific") else ["HIB19", "iso", "auto"]
    for reagent in reagents:
        for d in (DOSES if reagent != "auto" else [0.0]):
            cand = lay[(lay.reagent == reagent) & (lay.dose == d)]
            if len(cand) and not cand[col].notna().any():
                why = sorted({why_missing(r, needs_bf) for r in cand.itertuples()})
                if col.endswith("_specific") and why == ["n/a"]:
                    why = ["no iso"]
                notes.append((", ".join(why), reagent if reagent == "auto" else f"{reagent} {d:g} µg"))
    if notes:
        ax.text(1.0, 1.01, _notes_line(notes), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5, color=MUTED)
    ax.axvline(-1.5, color=GRID, lw=0.8, zorder=0)
    ax.set_xticks([X_AUTO, -1, 0, 1], labels=["auto", "0.5", "1", "2"])
    ax.set_xlim(-2.5, 1.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    v = sub[col]
    return (v.min(), v.max()) if len(v) else (np.nan, np.nan)


def _notes_line(pairs) -> str:
    """[(reason, label), ...] -> 'no BF: HIB19 2 µg, iso 0.5 µg; no fluo: ...' (one line)."""
    d: dict = {}
    for reason, lab in pairs:
        d.setdefault(reason, []).append(lab)
    return "; ".join(f"{k}: {', '.join(v)}" for k, v in d.items())


def _fmt_delta(v) -> str:
    if abs(v) >= 100:
        return f"Δ {v:+,.0f}"
    return f"Δ {v:+.2f}" if abs(v) < 1 else f"Δ {v:+.1f}"


def _fluo_legend(fig, y=1.0, extra=None):
    h = [Line2D([], [], color=REAGENT_COLOR[r], marker="o", lw=2 if r != "auto" else 0,
                markersize=5, label=REAGENT_LABEL[r]) for r in REAGENTS]
    h.append(Line2D([], [], color=MUTED, marker="o", lw=0, mfc="white", label="BSA (circle)"))
    h.append(Line2D([], [], color=MUTED, marker="s", lw=0, mfc="white", label="BSA + IVIG (square)"))
    h += extra or []
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 6), bbox_to_anchor=(0.5, y))


# =====================================================================================
# Figure 8 – fluo plate maps
# =====================================================================================
def _fmt_au(v):
    return f"{v / 1000:.1f}k" if abs(v) >= 1000 else f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}"


def _pct_norm(vals, lo_pct=2, hi_pct=98):
    """Percentile-clipped linear colour scale (robust to skewed / outlying wells)."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return mcolors.Normalize(0, 1), False
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-9
    return mcolors.Normalize(lo, hi), bool(v.min() < lo or v.max() > hi)


def auto_intensity_table(ctx: Ctx) -> pd.DataFrame:
    """Autofluorescence (auto wells, no antibody) in intensity terms: per well and block mean."""
    a = ctx.fw[ctx.fw.reagent == "auto"]
    rows = [dict(level="well", well=int(r.well), block=r.block, n_auto_wells=1, mi_median=r.mi_median,
                 ii_total=r.ii_total, n_bf_used=r.n_bf_used, int_per_bf_cell=r.int_per_bf_cell)
            for r in a.itertuples()]
    for b in BLOCKS:
        g = a[a.block == b]
        rows.append(dict(level="block_mean", well=np.nan, block=b, n_auto_wells=len(g),
                         mi_median=g.mi_median.mean(), ii_total=g.ii_total.mean(), n_bf_used=g.n_bf_used.mean(),
                         int_per_bf_cell=g.int_per_bf_cell.mean()))
    return pd.DataFrame(rows)


def _auto_note(ctx: Ctx, col: str, label: str) -> str:
    a = ctx.fw[ctx.fw.reagent == "auto"]
    parts = []
    for b in BLOCKS:
        v = a.loc[a.block == b, col].dropna()
        parts.append(f"{BLOCK_SHORT[b]}: {_fmt_au(v.mean())}" if len(v) else f"{BLOCK_SHORT[b]}: n/a")
    return f"Autofluorescence (auto-well mean), {label}: " + "  |  ".join(parts) + " a.u."


def _fmt_fold(v):
    return f"{v:.2f}"


def _auto_means(ctx: Ctx, col: str, fmt=_fmt_au) -> str:
    a = ctx.fw[ctx.fw.reagent == "auto"]
    return "  |  ".join(f"{BLOCK_SHORT[b]}: {fmt(a.loc[a.block == b, col].mean())}" for b in BLOCKS)


def _intensity_plates(ctx: Ctx, specs, name: str, footer_text: str, suptitle: str | None = None,
                      height: float = 7.8) -> None:
    """specs: (column, title, colourbar label, note under the plate, fmt, needs_bf[, text for missing wells])."""
    fw = ctx.fw
    out = (lambda r: r.bf_qc_status == "high" or r.fluo_qc_status == "high")
    fig, axes = plt.subplots(len(specs), 1, figsize=(7.4, height))
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, fontweight="bold", y=0.995)
    for ax, spec in zip(np.atleast_1d(axes), specs):
        col, title, cbl, note, fmt, needs_bf = spec[:6]
        miss = spec[6] if len(spec) > 6 else None
        v = fw.assign(present=fw.fluo_present & fw.bf_present) if needs_bf else fw.assign(present=fw.fluo_present)
        norm, clipped = _pct_norm(v.loc[v.present, col])
        mf = (lambda r, t=miss: t) if miss else (why_missing if needs_bf else (lambda r: "no fluo"))
        m = draw_plate(ax, v, col, fmt, title, flag_fn=flag2, outline_fn=out, missing_fn=mf, norm=norm)
        fig.colorbar(m, ax=ax, fraction=0.02, pad=0.14, extend="both" if clipped else "neither",
                     label=cbl + "\n(scale: P2–P98)")
        ax.text(0.0, 3.3, note, fontsize=5.8, color=MUTED, ha="left", va="top", clip_on=False)
    fig.subplots_adjust(hspace=0.6, top=0.93 if suptitle else 0.96)
    fluo_footer(fig, ctx, footer_text)
    save(fig, name, ctx)


def _ref_text(ctx: Ctx) -> str:
    return "  |  ".join(f"{BLOCK_SHORT[b]}: {ctx.thresholds[b]['mi_auto_ref']:.1f}" for b in BLOCKS)


def _fmt_M(v):
    return f"{v / 1e6:.0f}M" if abs(v) >= 1e6 else f"{v / 1e3:.0f}k"


def _fmt_2(v):
    return f"{v:.2f}"


def fig_fluo_plate(ctx: Ctx) -> None:
    """Figure 8: intensity-based plate maps, normalised by BF-segmented AREA (not by cell count)."""
    name = "autofluorescence_intensity" + ("_allobjects" if ctx.include_small else "") + ".csv"
    fw = ctx.fw
    a = fw[fw.reagent == "auto"]
    t = a[["well", "block", "mi_median", "mi_p90", "ii_total", "bf_area_qc", "fluo_area_qc", "ii_per_bf_area",
           "int_per_bf_cell", "excess_per_bf_cell"]].assign(level="well")
    m = a.groupby("block")[["mi_median", "mi_p90", "ii_total", "bf_area_qc", "fluo_area_qc", "ii_per_bf_area",
                            "int_per_bf_cell", "excess_per_bf_cell"]].mean().reset_index().assign(level="block_mean")
    m["n_auto_wells"] = m.block.map(a.groupby("block").size())
    pd.concat([t, m]).assign(mi_auto_ref_block_median=lambda d: d.block.map(
        {bb: ctx.thresholds[bb]["mi_auto_ref"] for bb in BLOCKS})).to_csv(FIG_DIR / name, index=False, float_format="%.6g")
    print(f"  saved {OUT_DIR}/{name}")
    common = ("Linear colour scale clipped at the 2nd–98th percentile of the wells shown (true values printed in the wells; colourbar "
              "arrows = clipped). Replicate wells are shown individually. Mean Intensity may be clipped at ~148 a.u.")
    area_def = ("BF area = Σ Area of QC BF objects (Area ≥ %d px²) of the well; Integrated fluorescence = Σ Integrated Intensity "
                "(= Mean Intensity × Area) of the QC fluo objects. Area normalisation is used because the BF cell COUNT is unreliable in "
                "confluent regions (merged objects)." % AREA_MIN_PX)
    _intensity_plates(ctx, [
        ("mi_median", "a  Median Mean Intensity per well (a.u.)", "Median Mean Intensity (a.u.)",
         "Autofluorescence (auto-well mean): " + _auto_means(ctx, "mi_median") + " a.u.", _fmt_au, False),
        ("ii_total", "b  Integrated fluorescence per well (no area normalization)",
         "Integrated fluorescence per well\n(a.u.; no area normalization)",
         "Autofluorescence (auto-well mean): " + _auto_means(ctx, "ii_total", _fmt_M) + " a.u. (M = 10⁶)", _fmt_M, False),
        ("ii_per_bf_area", "c  Integrated fluorescence per BF-segmented area (a.u. per px²)",
         "Integrated fluorescence /\nBF area (a.u. per px²)",
         "Autofluorescence (auto-well mean): " + _auto_means(ctx, "ii_per_bf_area", _fmt_2) + " a.u. per px²", _fmt_2, True),
        ("ii_per_bf_area_bgcorr", "d  Integrated fluorescence per BF area, background-corrected (a.u. per px²)",
         "Background-corrected\nintegrated fluorescence / BF area\n(a.u. per px²)",
         "Background = auto-well mean of the same block: " + _auto_means(ctx, "ii_per_bf_area", _fmt_2) + " a.u. per px²",
         _fmt_2, True),
        ("ii_per_bf_area_fold", "e  Integrated fluorescence per BF area, fold over autofluorescence",
         "Fold over auto-well mean\nof the same block",
         "Auto-well mean of the same block = 1.0 (BSA and BSA + IVIG separately).", _fmt_2, True),
    ], "fig8_fluo_plate_heatmap", area_def + "\n" + common, height=12.6)
    # supplementary: normalisation by BF cell COUNT
    _intensity_plates(ctx, [
        ("int_per_bf_cell", "a  Integrated fluorescence per BF cell (count), all objects (a.u.)",
         "Integrated fluorescence\nper BF cell (a.u.)",
         "Autofluorescence (auto-well mean): " + _auto_means(ctx, "int_per_bf_cell") + " a.u.", _fmt_au, True),
        ("int_per_bf_cell_bgcorr", "b  Same, background-corrected (minus auto-well mean of the block)",
         "Background-corrected\nintegrated fluorescence\nper BF cell (a.u.)",
         "Auto replicates 25/26 differ ~3x; the subtraction produces many negative wells.", _fmt_au, True),
        ("excess_per_bf_cell", "c  Excess integrated fluorescence per BF cell (a.u.)",
         "Excess integrated fluorescence\nper BF cell (a.u.)",
         "Excess = Σ max(Mean Intensity − MI_auto, 0) × Area; MI_auto = " + _ref_text(ctx) + " a.u.", _fmt_au, True),
        ("excess_fold", "d  Excess per BF cell, fold over autofluorescence", "Fold over auto-well mean\nof the same block",
         "Auto-well mean of the same block = 1.0.", _fmt_2, True),
    ], "fig8c_per_cell_count_supplementary",
        "Why this is supplementary: the BF cell count is unreliable in confluent regions (merged objects), and these readouts also scale "
        "with object count and area (area differs between blocks, ~550 vs ~1000 px²); auto replicates differ up to 3x.\n" + common,
        suptitle="Supplementary: per-BF-cell-count normalisation (count unreliable in confluent regions)", height=10.2)
    # intensity-only maps without any BF denominator
    ref = _ref_text(ctx)
    _intensity_plates(ctx, [
        ("delta_mi_median", "a  Δ median Mean Intensity vs. auto median of the block (a.u.)",
         "Median Mean Intensity −\nMI_auto (a.u.)",
         f"MI_auto (median Mean Intensity of all QC fluo objects in auto wells): {ref} a.u.", _fmt_au, False),
        ("mi_p90", "b  P90 of Mean Intensity per well (a.u.)", "P90 of Mean Intensity\n(a.u.)",
         "Autofluorescence (auto-well mean), P90: " + _auto_means(ctx, "mi_p90") + " a.u.", _fmt_au, False),
    ], "fig8d_mi_delta_p90_plate", "Both readouts use per-object Mean Intensity only (no counts, no BF denominator). " + common,
        height=5.6)
    fig_fluo_plate_counts(ctx)


def fig_fluo_plate_counts(ctx: Ctx) -> None:
    """Supplementary: the former count-based plate maps, kept for comparison only."""
    fw = ctx.fw
    out = (lambda r: r.bf_qc_status == "high" or r.fluo_qc_status == "high")
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.3))
    fig.suptitle("Supplementary: object-count-based maps (not used for autofluorescence correction)",
                 fontsize=9, fontweight="bold", y=0.99)
    v1 = fw.assign(present=fw.fluo_present)
    m = draw_plate(axes[0], v1, "n_fluo_used", _fmt_count, "a  Fluo objects per well", vmin=0,
                   flag_fn=flag2, outline_fn=out, missing_fn=lambda r: "no fluo")
    fig.colorbar(m, ax=axes[0], fraction=0.02, pad=0.14, label="Fluo objects\n(per well)")
    v2 = fw.assign(present=fw.fluo_present & fw.bf_present)
    m = draw_plate(axes[1], v2, "fluo_ratio", _fmt_ratio, "b  Fluo objects per BF cell", vmin=0,
                   flag_fn=flag2, outline_fn=out, missing_fn=why_missing)
    fig.colorbar(m, ax=axes[1], fraction=0.02, pad=0.14, label=f"Fluo objects per BF cell\n({fluo_sel_label(ctx)})")
    m = draw_plate(axes[2], v2, "bright_ratio", _fmt_small, "c  Bright fluo objects per BF cell (intensity-gated count)",
                   vmin=0, flag_fn=flag2, outline_fn=out, missing_fn=why_missing)
    fig.colorbar(m, ax=axes[2], fraction=0.02, pad=0.14, label=f"Bright objects per BF cell\n({fluo_sel_label(ctx)})")
    fig.subplots_adjust(hspace=0.32, top=0.91)
    fluo_footer(fig, ctx, "Counts depend on the fluo segmentation threshold, not on signal strength (auto wells also yield many "
                          "objects). b, c: ratio of two independent segmentations, not a fraction of positive cells; "
                          "* = ratio > 1.")
    save(fig, "fig8b_fluo_counts_supplementary", ctx)


# =====================================================================================
# Figure 9 – dose-response of fluo metrics (raw / background-corrected / specific)
# =====================================================================================
def _metric_grid(ctx: Ctx, kind: str, name: str, extra_note: str, keys=None) -> None:
    keys = list(keys or FLUO_METRICS)
    fig, axes = plt.subplots(len(keys), 2, figsize=(7.6, 2.45 * len(keys)), squeeze=False)
    for i, k in enumerate(keys):
        col = {"raw": k, "bgcorr": f"{k}_bgcorr", "specific": f"{k}_specific"}[kind]
        ranges = [draw_metric_dose(axes[i, j], ctx, b, col, k, ref="auto" if kind == "raw" else "zero")
                  for j, b in enumerate(BLOCKS)]
        _shared_ylim(axes[i], ranges, zero_floor=(kind == "raw"))
        axes[i, 0].set_ylabel(metric_label(k, ctx, kind), fontsize=6.5)
        plt.setp(axes[i, 1].get_yticklabels(), visible=False)
        for j in range(2):
            if i == len(keys) - 1:
                axes[i, j].set_xlabel(f"Antibody dose ({DOSE_UNIT})")
    for j, b in enumerate(BLOCKS):
        axes[0, j].set_title(f"Block: {b}", pad=14)
    fig.tight_layout(h_pad=2.0, w_pad=0.6)
    if kind == "specific":
        fig.legend(handles=[Line2D([], [], color=REAGENT_COLOR["HIB19"], marker="o", lw=2, markersize=5,
                                   label="HIB19 − isotype (dot = replicate, line = mean)"),
                            Line2D([], [], color=MUTED, marker="o", lw=0, mfc="white", label="BSA (circle)"),
                            Line2D([], [], color=MUTED, marker="s", lw=0, mfc="white", label="BSA + IVIG (square)")],
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    else:
        extra = [Line2D([], [], color=MUTED, ls=":", lw=1, label="auto mean")] if kind == "raw" else None
        _fluo_legend(fig, y=1.0, extra=extra)
    fluo_footer(fig, ctx, extra_note)
    save(fig, name, ctx)


def fig_fluo_dose(ctx: Ctx) -> None:
    _metric_grid(ctx, "raw", "fig9_fluo_dose_response",
                 "Dots: wells (number next to dot); line and bar: mean of replicate wells. Light→dark = 0.5→2 µg.")
    _metric_grid(ctx, "bgcorr", "fig9_fluo_dose_response_bgcorr",
                 "Background-corrected: value minus the mean of the auto wells of the same block (n per block in "
                 "well_summary_combined.csv, <metric>_auto_n).")
    _metric_grid(ctx, "specific", "fig9b_fluo_specific_signal",
                 "Specific signal = HIB19 well minus the isotype well of the same block, dose and replicate "
                 "(iso mean if that well is missing). Each dot is one HIB19 replicate.")


# =====================================================================================
# Figure 10 – Fc block effect (BSA vs BSA + IVIG), paired by plate column
# =====================================================================================
D_OFF = {0.5: -0.14, 1.0: 0.0, 2.0: 0.14, 0.0: 0.0}


def draw_fc_panel(ax, ctx: Ctx, reagent: str, col: str, metric: str, label=True):
    fw = ctx.fw
    needs_bf = metric in METRIC_NEEDS_BF
    lo, hi, notes, deltas = [], [], [], []
    for d in (DOSES if reagent != "auto" else [0.0]):
        c, ec = well_color(reagent, d), REAGENT_COLOR[reagent]
        grp = {b: fw[(fw.reagent == reagent) & (fw.dose == d) & (fw.block == b)].sort_values("replicate")
               for b in BLOCKS}
        xb = {BLOCK_BSA: 0 + D_OFF[d], BLOCK_IVIG: 1 + D_OFF[d]}
        for rep in (1, 2):
            vals = {}
            for b in BLOCKS:
                r = grp[b][grp[b].replicate == rep]
                vals[b] = r.iloc[0] if len(r) else None
            ys = {b: (getattr(vals[b], col) if vals[b] is not None else np.nan) for b in BLOCKS}
            if all(pd.notna(ys[b]) for b in BLOCKS):
                ax.plot([xb[BLOCK_BSA], xb[BLOCK_IVIG]], [ys[BLOCK_BSA], ys[BLOCK_IVIG]], color=ec, lw=0.7,
                        alpha=0.55, zorder=2)
            for b in BLOCKS:
                if pd.notna(ys[b]):
                    ax.scatter(xb[b], ys[b], s=22, marker=BLOCK_MARKER[b], color=c, edgecolor=ec, lw=0.9, zorder=3)
                    if label:
                        ax.annotate(f"{vals[b].well}{flag2(vals[b])}", (xb[b], ys[b]), fontsize=4.8, color=MUTED,
                                    xytext=(-3 if b == BLOCK_BSA else 3, 0), textcoords="offset points",
                                    ha="right" if b == BLOCK_BSA else "left", va="center")
                    lo.append(ys[b])
                    hi.append(ys[b])
        means = {b: grp[b][col].mean() for b in BLOCKS}
        for b in BLOCKS:
            if pd.notna(means[b]):
                ax.scatter(xb[b], means[b], marker="_", s=110, color=ec, lw=2, zorder=2.5)
            else:
                why = sorted({why_missing(r, needs_bf) for r in grp[b].itertuples()}) or ["not in layout"]
                notes.append((", ".join(why), f"{BLOCK_SHORT[b]}" if reagent == "auto"
                              else f"{d:g} µg {BLOCK_SHORT[b]}"))
        if all(pd.notna(means[b]) for b in BLOCKS):
            ax.plot([xb[BLOCK_BSA], xb[BLOCK_IVIG]], [means[BLOCK_BSA], means[BLOCK_IVIG]], color=ec, lw=2.2,
                    zorder=2.4)
            dlt = means[BLOCK_IVIG] - means[BLOCK_BSA]
            deltas.append(("" if reagent == "auto" else f"{d:g} µg: ") + _fmt_delta(dlt).replace("Δ ", ""))
    if deltas:
        # legend with loc="best" so the Δ box avoids the data points
        leg = ax.legend(handles=[Line2D([], [], lw=0, label=s) for s in deltas], title="Δ (IVIG − BSA), means",
                        loc="best", handlelength=0, handletextpad=0, fontsize=5.3, title_fontsize=5.3,
                        frameon=True, framealpha=0.9, edgecolor=GRID, borderpad=0.4, labelspacing=0.2)
        leg.set_zorder(5)
    if notes:
        ax.text(1.0, 1.01, _notes_line(notes), transform=ax.transAxes, ha="right", va="bottom", fontsize=5,
                color=MUTED)
    ax.set_xticks([0, 1], labels=["BSA", "BSA + IVIG"])
    ax.set_xlim(-0.45, 1.6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    return (min(lo), max(hi)) if lo else (np.nan, np.nan)


def _fc_figure(ctx: Ctx, kind: str, name: str, keys=None) -> None:
    keys = list(keys or FLUO_METRICS)
    reagents = REAGENTS if kind == "raw" else ["HIB19", "iso"]
    fig, axes = plt.subplots(len(keys), len(reagents), figsize=(2.65 * len(reagents) + 1.2, 2.5 * len(keys)),
                             squeeze=False)
    for i, k in enumerate(keys):
        col = k if kind == "raw" else f"{k}_bgcorr"
        ranges = [draw_fc_panel(axes[i, j], ctx, r, col, k) for j, r in enumerate(reagents)]
        _shared_ylim(axes[i], ranges, zero_floor=(kind == "raw"))
        if kind != "raw":
            for ax in axes[i]:
                ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        axes[i, 0].set_ylabel(metric_label(k, ctx, kind), fontsize=6.5)
        for ax in axes[i, 1:]:
            plt.setp(ax.get_yticklabels(), visible=False)
    for j, r in enumerate(reagents):
        axes[0, j].set_title(REAGENT_LABEL[r], pad=14)
    fig.tight_layout(h_pad=2.0, w_pad=0.5)
    h = [Patch(facecolor=well_color("HIB19", d), edgecolor=REAGENT_COLOR["HIB19"], label=f"{d:g} µg") for d in DOSES]
    h += [Line2D([], [], color=MUTED, lw=0.7, label="replicate pair (same plate column)"),
          Line2D([], [], color=MUTED, lw=2.2, label="mean of replicates"),
          Line2D([], [], color=MUTED, marker="o", lw=0, mfc="white", label="BSA"),
          Line2D([], [], color=MUTED, marker="s", lw=0, mfc="white", label="BSA + IVIG")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fluo_footer(fig, ctx, "Each well in row A (BSA) is paired with the well in the same column of row B (BSA + IVIG); "
                          "auto: 25↔27, 26↔28. Δ = mean(IVIG) − mean(BSA)." +
                ("" if kind == "raw" else "  Values are background-corrected with the auto wells of the SAME block."))
    save(fig, name, ctx)


def fig_fc_block(ctx: Ctx) -> None:
    _fc_figure(ctx, "raw", "fig10_fc_block_effect")
    _fc_figure(ctx, "bgcorr", "fig10b_fc_block_effect_bgcorr")


def fig_fluo_dose_v2(ctx: Ctx) -> None:
    """Copy of fig9 with intensity-based metrics only (median MI, P90 MI, excess per BF cell, fold over auto)."""
    k = list(FLUO_METRICS_V2)
    ref = _ref_text(ctx)
    _metric_grid(ctx, "raw", "fig9_v2_fluo_dose_response",
                 f"Intensity-based metrics only. Excess = Σ max(Mean Intensity − MI_auto, 0) × Area per object, MI_auto = auto-well "
                 f"median per block ({ref} a.u.). Dots: wells; line: mean of replicates.", keys=k)
    _metric_grid(ctx, "bgcorr", "fig9_v2_fluo_dose_response_bgcorr",
                 "Value minus the mean of the auto wells of the same block (for the fold metric: fold − 1).", keys=k)
    _metric_grid(ctx, "specific", "fig9b_v2_fluo_specific_signal",
                 "Specific signal = HIB19 well minus isotype well of the same block, dose and replicate.", keys=k)
    compare_background_metrics(ctx)


def fig_fc_block_v2(ctx: Ctx) -> None:
    _fc_figure(ctx, "raw", "fig10_v2_fc_block_effect", keys=list(FLUO_METRICS_V2))
    _fc_figure(ctx, "bgcorr", "fig10b_v2_fc_block_effect_bgcorr", keys=list(FLUO_METRICS_V2))


COMPARE_METRICS = ["fluo_ratio", "bright_ratio", "int_per_bf_cell", "mi_median", "mi_p90", "delta_mi_median",
                   "excess_per_bf_cell", "excess_fold"]


def compare_background_metrics(ctx: Ctx) -> None:
    """Count-/all-object-based (v1) vs intensity-based (v2) metrics: replicate agreement and sign of the
    background-corrected value. CSV only."""
    fw = ctx.fw[ctx.fw.fluo_present]
    rows = []
    for m in COMPARE_METRICS:
        d = []
        for _, g in fw.groupby(["block", "reagent", "dose"]):
            v = g[m].dropna().to_numpy(float)
            if len(v) == 2 and np.mean(np.abs(v)) > 0:
                d.append(abs(v[0] - v[1]) / np.mean(np.abs(v)))
        auto = fw[fw.reagent == "auto"][m].dropna().to_numpy(float)
        bg = f"{m}_bgcorr"
        non = fw[fw.reagent != "auto"][bg].dropna() if bg in fw.columns else pd.Series(dtype=float)
        v2 = fw[m].dropna()
        rows.append(dict(
            metric=m, family="v2 (intensity)" if m in list(FLUO_METRICS_V2) + ["delta_mi_median"] else "v1 (count / all-object)",
            median_replicate_rel_diff=float(np.median(d)) if d else np.nan,
            max_replicate_rel_diff=float(np.max(d)) if d else np.nan,
            auto_replicate_rel_diff=float(abs(auto[0] - auto[1]) / np.mean(np.abs(auto))) if False else np.nan,
            frac_nonauto_wells_negative_bgcorr=float((non < 0).mean()) if len(non) else np.nan,
            n_wells=int(len(v2))))
    # auto replicate agreement within block
    for r in rows:
        dd = []
        for _, g in fw[fw.reagent == "auto"].groupby("block"):
            v = g[r["metric"]].dropna().to_numpy(float)
            if len(v) == 2 and np.mean(np.abs(v)) > 0:
                dd.append(abs(v[0] - v[1]) / np.mean(np.abs(v)))
        r["auto_replicate_rel_diff_mean"] = float(np.mean(dd)) if dd else np.nan
    out = pd.DataFrame(rows).drop(columns=["auto_replicate_rel_diff"])
    name = "compare_background_metrics" + ("_allobjects" if ctx.include_small else "") + ".csv"
    out.to_csv(FIG_DIR / name, index=False, float_format="%.4g")
    print(f"  saved {OUT_DIR}/{name}")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


# =====================================================================================
# Figure 11 – fluo morphology / intensity distributions and QC
# =====================================================================================
def _fluo_view(ctx: Ctx):
    v = ctx.fw.copy()
    v["present"] = v.fluo_present
    v["qc_status"] = v.fluo_qc_status

    def values(feature, well):
        g = ctx.fby_well.get(well)
        return None if g is None or len(g) == 0 else g[feature].to_numpy(float)
    return SimpleNamespace(wells=v, cells=ctx.fl, values=values)


def _per_well_bars(ax, ctx: Ctx, col: str, title: str, ylabel: str, fmt=_fmt_small):
    fw = ctx.fw
    ymax = np.nanmax(fw[col].to_numpy(float)) if fw[col].notna().any() else 1.0
    for r in fw.itertuples():
        v = getattr(r, col)
        if pd.notna(v):
            ax.bar(r.well, v, width=0.75, color=well_color(r.reagent, r.dose), edgecolor=REAGENT_COLOR[r.reagent],
                   lw=0.6)
            ax.text(r.well, v + ymax * 0.01, fmt(v), ha="center", va="bottom", fontsize=4.6, rotation=90)
        else:
            ax.bar(r.well, ymax * 0.04, width=0.75, color=NODATA, hatch="////", edgecolor=HATCH, lw=0)
            ax.text(r.well, ymax * 0.05, why_missing(r, col != "mi_median"), ha="center", va="bottom", fontsize=4.5,
                    color=MUTED, rotation=90)
    for b, ls in zip(BLOCKS, ["--", ":"]):
        a = fw[(fw.reagent == "auto") & (fw.block == b)][col].dropna()
        if len(a):
            ax.axhline(a.mean(), color=REAGENT_COLOR["auto"], ls=ls, lw=1,
                       label=f"auto mean {BLOCK_SHORT[b]} (n = {len(a)})")
    ax.set_xticks(fw.well, labels=[f"{w}" for w in fw.well], fontsize=5.5)
    ax.set_xlim(0.4, len(fw) + 0.6)
    ax.set_ylim(0, ymax * 1.22)
    ax.set_title(title)
    ax.set_ylabel(ylabel, fontsize=6.5)
    ax.set_xlabel("Well")
    ax.legend(loc="upper left", fontsize=5.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_fluo_morph_qc(ctx: Ctx) -> None:
    view = _fluo_view(ctx)
    feats = ["Area", "Circularity", "Eccentricity", MI_COL]
    widths = [len(block_positions(ctx.wells, b)) + 1.4 for b in BLOCKS]
    fig = plt.figure(figsize=(10.5, 13.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.25], hspace=0.22)
    top = outer[0].subgridspec(len(feats), 2, width_ratios=widths, hspace=0.12, wspace=0.06)
    rng = np.random.default_rng(SEED)
    first = None
    for i, f in enumerate(feats):
        yl = feature_ylim(view, f)
        for j, b in enumerate(BLOCKS):
            ax = fig.add_subplot(top[i, j], sharey=first if j else None)
            if j == 0:
                first = ax
            draw_violins(ax, view, b, f, yl, label_x=(i == len(feats) - 1), rng=rng)
            if i == 0:
                ax.set_title(f"Block: {b}", pad=10)
            if j == 0:
                ax.set_ylabel(f"Fluo object {f} (a.u.)" if f == MI_COL else f"Fluo object {axis_label(f)}",
                              fontsize=7)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
        first = None
    bot = outer[1].subgridspec(1, 3, width_ratios=[1, 1.35, 1.35], wspace=0.3)
    ax = fig.add_subplot(bot[0, 0])
    fw = ctx.fw[ctx.fw.fluo_present]
    for r in fw.itertuples():
        ax.scatter(r.n_fluo_total, r.fluo_fraction_small * 100, s=26, marker=BLOCK_MARKER[r.block],
                   color=well_color(r.reagent, r.dose), edgecolor=REAGENT_COLOR[r.reagent], lw=0.9, zorder=3)
        ax.annotate(str(r.well), (r.n_fluo_total, r.fluo_fraction_small * 100), fontsize=5, xytext=(3, 2),
                    textcoords="offset points")
    ax.axhline(FLUO_HIGH_SMALL_FRAC * 100, color=STATUS_COLOR["high"], ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fluo objects per well (all, log scale)")
    ax.set_ylabel(f"Fluo objects with Area < {FLUO_AREA_MIN_PX} px² (%)", fontsize=6.5)
    ax.set_title("e  Fluo count vs. small-object fraction")
    ax.grid(color=GRID, lw=0.6)
    _per_well_bars(fig.add_subplot(bot[0, 1]), ctx, "fluo_ratio", "f  Segmentation only (noise floor = auto)",
                   f"Fluo objects per BF cell\n({fluo_sel_label(ctx)})")
    _per_well_bars(fig.add_subplot(bot[0, 2]), ctx, "bright_ratio", "g  Intensity-gated (bright objects)",
                   f"Bright fluo objects per BF cell\n({fluo_sel_label(ctx)})")
    fig.text(0.1, 0.905, "a–d  Distributions of fluo objects per well", fontsize=9, fontweight="bold")
    _dose_color_legend(fig, y=0.065)
    fluo_footer(fig, ctx, f"Violins: density of up to {VIOLIN_MAX_CELLS:,} sampled objects per well; dot = median, "
                          "bar = IQR; y-axis limited to the 0.5–99.5th percentile. f/g: auto wells (no antibody) "
                          "define the noise floor; dashed/dotted lines = auto mean per block.", y=0.045)
    save(fig, "fig11_fluo_morphology_qc", ctx)
    fig_fluo_intensity_dist(ctx)
    fig_area_vs_intensity(ctx)


def _gate_sensitivity_note(ctx: Ctx) -> str:
    sel = ctx.fw[(ctx.fw.reagent == "HIB19") & (ctx.fw.dose == 2.0) & ctx.fw.fluo_present]
    parts = "; ".join(f"{BLOCK_SHORT[r.block]} well {r.well}: {r.frac_bright:.0%}" for r in sel.itertuples())
    return ("Note: the P99(auto) gate cuts through the main intensity population in HIB19 2 µg wells (share of objects above "
            f"the gate – {parts}), so bright fractions are gate-sensitive.")


def fig_fluo_intensity_dist(ctx: Ctx) -> None:
    """Per-well Mean Intensity: histogram on log bins (bimodality check) and the fraction of
    objects brighter than x (complementary ECDF, log-log: shows bright tails)."""
    fw = ctx.fw
    mi_all = ctx.fl[MI_COL].to_numpy(float)
    bins = np.geomspace(max(np.nanmin(mi_all), 1e-3), np.nanmax(mi_all) * 1.01, 60)
    centers = np.sqrt(bins[:-1] * bins[1:])
    fig, axes = plt.subplots(4, 3, figsize=(10, 10.5), sharex=True)
    for bi, b in enumerate(BLOCKS):
        for di, d in enumerate(DOSES):
            ah, ac = axes[2 * bi, di], axes[2 * bi + 1, di]
            sel = fw[(fw.block == b) & (((fw.reagent != "auto") & (fw.dose == d)) | (fw.reagent == "auto"))]
            missing = []
            for r in sel.sort_values(["reagent", "replicate"]).itertuples():
                g = ctx.fby_well.get(r.well)
                if g is None or len(g) == 0:
                    missing.append(f"{r.well} ({r.reagent})")
                    continue
                v = g[MI_COL].to_numpy(float)
                ls = "-" if r.replicate == 1 else "--"
                c = REAGENT_COLOR[r.reagent]
                h, _ = np.histogram(v, bins)
                ah.plot(centers, h / h.sum() * 100, color=c, ls=ls, lw=1.1, label=f"{r.well} {r.reagent}")
                s = np.sort(v)
                surv = 1.0 - np.arange(1, len(s) + 1) / len(s)
                ac.step(s[:-1], surv[:-1], where="post", color=c, ls=ls, lw=1.1)
            for ax in (ah, ac):
                ax.axvline(ctx.thresholds[b]["primary"], color=INK, lw=0.9)
                ax.axvline(ctx.thresholds[b]["iso"], color=INK, lw=0.9, ls=":")
                ax.set_xscale("log")
                ax.grid(color=GRID, lw=0.5)
            ac.set_yscale("log")
            ac.set_ylim(1e-4, 1.2)
            ah.set_title(f"{BLOCK_SHORT[b]} – {d:g} µg (+ auto)", fontsize=8)
            ah.legend(fontsize=5, loc="upper right", ncol=1)
            if missing:
                ah.text(0.02, 0.97, "no data: " + ", ".join(missing), transform=ah.transAxes, fontsize=5,
                        color=MUTED, va="top")
            if di == 0:
                ah.set_ylabel("Objects per bin (% of well)", fontsize=6.5)
                ac.set_ylabel("Fraction of objects\nwith Mean Intensity > x", fontsize=6.5)
    for ax in axes[-1]:
        ax.set_xlabel("Mean Intensity of fluo object (a.u., log scale)")
    fig.tight_layout(h_pad=1.0)
    fluo_footer(fig, ctx, "Solid = replicate 1, dashed = replicate 2. Vertical lines: intensity gate (solid = "
                          f"P{FLUO_INT_PERCENTILE:g} of auto, dotted = P{FLUO_INT_PERCENTILE:g} of iso, same block).\n"
                          + _gate_sensitivity_note(ctx))
    save(fig, "fig11b_fluo_intensity_distributions", ctx)


def fig_area_vs_intensity(ctx: Ctx) -> None:
    fw = ctx.fw
    conds = [(r, d) for r in ["HIB19", "iso"] for d in DOSES] + [("auto", 0.0)]
    fig, axes = plt.subplots(2, len(conds), figsize=(13, 4.6), sharex=True, sharey=True)
    rng = np.random.default_rng(SEED)
    amax = np.nanpercentile(ctx.fl["Area"], 99.5)
    ylo, yhi = np.nanmin(ctx.fl[MI_COL]), np.nanmax(ctx.fl[MI_COL])
    for bi, b in enumerate(BLOCKS):
        for ci, (reag, d) in enumerate(conds):
            ax = axes[bi, ci]
            cand = fw[(fw.block == b) & (fw.reagent == reag) & (fw.dose == d) & fw.fluo_present].sort_values("replicate")
            if cand.empty:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=NODATA, hatch="////",
                                       edgecolor=HATCH, lw=0))
                ax.text(.5, .5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=6, color=MUTED)
                ax.set_title(f"{reag} {'' if reag == 'auto' else f'{d:g} µg'}", fontsize=6.5)
                continue
            r = cand.iloc[0]
            g = ctx.fby_well[int(r.well)]
            idx = subsample(np.arange(len(g)), 20_000, rng)
            ax.hexbin(g["Area"].to_numpy(float)[idx], g[MI_COL].to_numpy(float)[idx], gridsize=35, bins="log",
                      yscale="log", cmap=SEQ_CMAP, mincnt=1, linewidths=0, extent=(0, amax, np.log10(ylo), np.log10(yhi)),
                      rasterized=True)
            ax.axhline(ctx.thresholds[b]["primary"], color=INK, lw=0.8)
            ax.set_title(f"well {r.well}{flag2(r)}: {reag} {'' if reag == 'auto' else f'{d:g} µg'}", fontsize=6.5)
            ax.set_xlim(0, amax)
            ax.set_yscale("log")
        axes[bi, 0].set_ylabel(f"{BLOCK_SHORT[b]}\nMean Intensity (a.u.)", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel(f"Fluo object {axis_label('Area')}", fontsize=6.5)
    fig.tight_layout()
    fluo_footer(fig, ctx, "One representative well per condition (replicate 1, or 2 if 1 is missing); up to 20,000 "
                          f"objects (seed {SEED}); hexagon shade = log object count, scaled separately in each panel (no shared colourbar); "
                          "line = intensity gate.\n" + _gate_sensitivity_note(ctx))
    save(fig, "fig11c_fluo_area_vs_intensity", ctx)


# =====================================================================================
# BF-to-fluo matching (exploratory) + Figure 12
# =====================================================================================
def compute_matching(ctx: Ctx) -> dict:
    """Registration (mutual nearest neighbours) and nearest-BF-cell matching.

    Always uses QC-filtered objects in pixel coordinates (independent of --include-small)."""
    if ctx.match is not None:
        return ctx.match
    from scipy.spatial import cKDTree
    bf = ctx.cells_all[~ctx.cells_all["is_small"]]
    fl = ctx.fl_all[~ctx.fl_all["is_small"]]
    R = float(MATCH_RADIUS_PX) if MATCH_RADIUS_PX else 0.5 * float(bf["Equivalent Diameter"].median())
    rows, cells = [], []
    for w in sorted(set(bf.well.unique()) & set(fl.well.unique())):
        b, f = bf[bf.well == w], fl[fl.well == w]
        B = b[["Centroid X", "Centroid Y"]].to_numpy(float)
        F = f[["Centroid X", "Centroid Y"]].to_numpy(float)
        tb, tf = cKDTree(B), cKDTree(F)
        d1, i1 = tb.query(F, distance_upper_bound=MATCH_PAIR_SEARCH_PX)
        fi = np.flatnonzero(np.isfinite(d1))
        bi = i1[fi]
        _, back = tf.query(B[bi])
        mut = back == fi
        fi, bi = fi[mut], bi[mut]
        dx, dy = F[fi, 0] - B[bi, 0], F[fi, 1] - B[bi, 1]
        n_pairs = len(fi)
        if n_pairs:
            off = np.array([np.median(dx), np.median(dy)])
            mad = np.array([np.median(np.abs(dx - off[0])), np.median(np.abs(dy - off[1]))])
            med_dist = float(np.median(np.hypot(dx, dy)))
        else:
            off, mad, med_dist = np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.nan
        reg_ok = bool(n_pairs >= MATCH_MIN_PAIRS and np.hypot(*off) <= R / 2 and np.nanmax(mad) <= R / 2)
        Fc = F - np.nan_to_num(off)
        d, j = tb.query(Fc, distance_upper_bound=R)
        m = np.isfinite(d)
        bright = f["bright"].to_numpy(bool)
        mi = f[MI_COL].to_numpy(float)
        carried = np.full(len(B), np.nan)
        np.fmax.at(carried, j[m], mi[m])
        pos = np.zeros(len(B), bool)
        pos[j[m & bright]] = True
        anyp = np.zeros(len(B), bool)
        anyp[j[m]] = True
        rows.append(dict(well=int(w), n_bf_qc=len(B), n_fluo_qc=len(F), n_bright=int(bright.sum()),
                         n_pairs=n_pairs, offset_x=off[0], offset_y=off[1], mad_x=mad[0], mad_y=mad[1],
                         median_pair_dist=med_dist, registration_ok=reg_ok,
                         match_rate_all=float(m.mean()),
                         match_rate_bright=float((m & bright).sum() / bright.sum()) if bright.sum() else np.nan,
                         frac_bf_with_fluo=float(anyp.mean()), frac_bf_positive=float(pos.mean())))
        cells.append(pd.DataFrame({"well": int(w), "Area": b["Area"].to_numpy(float),
                                   "Circularity": b["Circularity"].to_numpy(float),
                                   "matched_mi": carried, "positive": pos, "any_fluo": anyp}))
    reg = pd.DataFrame(rows)
    if reg.empty:
        ctx.match = dict(R=R, reg=reg, cells=pd.DataFrame(), reliable=False,
                         reasons=["no well with both BF and fluo data"], spread=(np.nan, np.nan))
        return ctx.match
    reg = reg.merge(ctx.wells[["well", "block", "reagent", "dose", "replicate"]], on="well")
    reasons = []
    auto = reg[reg.reagent == "auto"]
    if auto.empty:
        reasons.append("no auto well with both modalities (sanity check impossible)")
    elif (auto.frac_bf_positive > MATCH_MAX_AUTO_POS).any():
        reasons.append(f"auto wells have > {MATCH_MAX_AUTO_POS:.0%} BF cells with a bright partner")
    h = reg[reg.reagent == "HIB19"].match_rate_bright.dropna()
    if h.empty or h.median() < MATCH_MIN_BRIGHT_RATE:
        reasons.append(f"median match rate of bright objects in HIB19 wells "
                       f"{(h.median() if len(h) else np.nan):.2f} < {MATCH_MIN_BRIGHT_RATE}")
    if reg.registration_ok.mean() < 0.8:
        reasons.append(f"registration unstable in {int((~reg.registration_ok).sum())} of {len(reg)} wells")
    spread = (float(reg.offset_x.std()), float(reg.offset_y.std()))
    ctx.match = dict(R=R, reg=reg, cells=pd.concat(cells, ignore_index=True), reliable=not reasons,
                     reasons=reasons, spread=spread)
    reg.to_csv(FIG_DIR / "bf_fluo_matching.csv", index=False, float_format="%.6g")
    print(f"  matching radius R = {R:.1f} px; registration offsets across wells: SD x = {spread[0]:.2f}, "
          f"SD y = {spread[1]:.2f} px ({'consistent' if max(spread) <= R / 2 else 'INCONSISTENT'})")
    print(f"  auto wells: BF cells with ANY fluo partner {fmt_pct(auto.frac_bf_with_fluo)}, with a BRIGHT partner "
          f"{fmt_pct(auto.frac_bf_positive)}")
    print("  matching " + ("RELIABLE" if not reasons else "NOT reliable: " + "; ".join(reasons)))
    print(f"  saved {OUT_DIR}/bf_fluo_matching.csv")
    return ctx.match


def fmt_pct(s) -> str:
    s = pd.Series(s).dropna()
    return "–" if s.empty else f"{s.min():.1%}–{s.max():.1%}"


SIZE_BINS = [300, 600, 900, 1200, 1500, 2000, 3000, np.inf]


def _posfrac_by_size(cells: pd.DataFrame, min_n: int = 50) -> pd.DataFrame:
    c = cells.assign(sbin=pd.cut(cells.Area, SIZE_BINS, right=False))
    g = c.groupby(["well", "sbin"], observed=True).positive.agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= min_n]
    g["x"] = g.sbin.map(lambda iv: np.sqrt(iv.left * (iv.right if np.isfinite(iv.right) else iv.left * 1.5)))
    return g


def draw_posfrac_size(ax, ctx: Ctx, block: str, mt: dict):
    pf = _posfrac_by_size(mt["cells"])
    pf = pf.merge(ctx.wells[["well", "block", "reagent", "dose", "replicate"]], on="well")
    for r_ in REAGENTS:
        for (w, rep, d), g in pf[(pf.block == block) & (pf.reagent == r_)].groupby(["well", "replicate", "dose"]):
            ax.plot(g.x, g["mean"] * 100, color=well_color(r_, d), lw=1.3, ls="-" if rep == 1 else "--",
                    marker=BLOCK_MARKER[block], ms=3, mec=REAGENT_COLOR[r_], label=f"{w}")
    ax.set_xscale("log")
    ax.set_xlabel(f"BF cell {axis_label('Area')} (bin centre)")
    ax.set_ylabel("BF cells with a bright\nfluo partner (%)", fontsize=6.5)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(f"Positive fraction vs. BF cell size – {BLOCK_SHORT[block]}", fontsize=8)


def fig_matching(ctx: Ctx) -> None:
    mt = compute_matching(ctx)
    if not mt["reliable"]:
        print("  fig12 skipped – matching not reliable: " + "; ".join(mt["reasons"]))
        return
    reg = mt["reg"].sort_values("well")
    R = mt["R"]
    fig, axes = plt.subplots(4, 2, figsize=(10, 13))
    ax = axes[0, 0]
    ax.errorbar(reg.well - 0.12, reg.offset_x, yerr=reg.mad_x, fmt="o", ms=3.5, color=INK, lw=0.8, label="Δx")
    ax.errorbar(reg.well + 0.12, reg.offset_y, yerr=reg.mad_y, fmt="s", ms=3.5, color=MUTED, lw=0.8, label="Δy")
    ax.axhspan(-R / 2, R / 2, color=GRID, zorder=0, label=f"± R/2 = {R / 2:.1f} px")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("Well")
    ax.set_ylabel("Fluo − BF centroid offset (px)\nmedian ± MAD of mutual NN pairs", fontsize=6.5)
    ax.set_title("a  Registration offset per well")
    ax.legend(fontsize=6, loc="upper right", ncol=3)
    ax = axes[0, 1]
    ax.bar(reg.well - 0.2, reg.match_rate_all * 100, width=0.4, color="#BFBFBF", label="all fluo objects")
    ax.bar(reg.well + 0.2, reg.match_rate_bright * 100, width=0.4, color=INK, label="bright fluo objects")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Well")
    ax.set_ylabel(f"Fluo objects with a BF cell\nwithin R = {R:.1f} px (%)", fontsize=6.5)
    ax.set_title("b  Match rate")
    ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.set_title("b  Match rate", pad=16)
    ax = axes[1, 0]
    for r in reg.itertuples():
        ax.bar(r.well, r.frac_bf_positive * 100, width=0.75, color=well_color(r.reagent, r.dose),
               edgecolor=REAGENT_COLOR[r.reagent], lw=0.6)
        ax.text(r.well, r.frac_bf_positive * 100, f"{r.frac_bf_positive * 100:.1f}", ha="center", va="bottom",
                fontsize=4.8, rotation=90)
    ax.axhline(MATCH_MAX_AUTO_POS * 100, color=STATUS_COLOR["high"], ls="--", lw=0.8,
               label=f"sanity limit for auto wells ({MATCH_MAX_AUTO_POS:.0%})")
    ax.set_xlabel("Well")
    ax.set_ylabel("BF cells with a bright\nfluo partner (%)", fontsize=6.5)
    ax.set_title("c  Fluo-positive BF cells per well")
    ax.legend(fontsize=6, loc="upper left")
    for j, b in enumerate(BLOCKS):
        draw_posfrac_size(axes[1, 1] if j == 0 else axes[2, 0], ctx, b, mt)
    for a, letter in ((axes[1, 1], "d"), (axes[2, 0], "e")):
        t = a.get_title(loc="left") or a.get_title()
        a.set_title("", loc="center")
        a.set_title(f"{letter}  {t}", fontsize=9, loc="left")
    ylim = max(axes[1, 1].get_ylim()[1], axes[2, 0].get_ylim()[1])
    for ax in (axes[1, 1], axes[2, 0]):
        ax.set_ylim(0, ylim)
    c = mt["cells"].merge(ctx.wells[["well", "reagent"]], on="well")
    hib = c[(c.reagent == "HIB19") & c.matched_mi.notna()]
    ax = axes[2, 1]
    if len(hib):
        ax.hexbin(hib.Area, hib.matched_mi, gridsize=45, bins="log", yscale="log", cmap=SEQ_CMAP, mincnt=1,
                  linewidths=0, rasterized=True, extent=(300, np.percentile(hib.Area, 99.5),
                                                         np.log10(hib.matched_mi.min()), np.log10(hib.matched_mi.max())))
        ax.set_yscale("log")
    ax.set_xlabel(f"BF cell {axis_label('Area')}")
    ax.set_ylabel("Mean Intensity of matched\nfluo object (a.u.)", fontsize=6.5)
    ax.set_title("f  HIB19 wells: BF area vs. matched intensity")
    ax = axes[3, 0]
    if len(hib):
        ax.hexbin(hib.Circularity.clip(upper=1), hib.matched_mi, gridsize=45, bins="log", yscale="log", cmap=SEQ_CMAP,
                  mincnt=1, linewidths=0, rasterized=True)
        ax.set_yscale("log")
    ax.set_xlabel("BF cell Circularity (unitless)")
    ax.set_ylabel("Mean Intensity of matched\nfluo object (a.u.)", fontsize=6.5)
    ax.set_title("g  HIB19 wells: BF circularity vs. matched intensity")
    for t in BLOCKS:
        for ax in (axes[2, 1], axes[3, 0]):
            ax.axhline(ctx.thresholds[t]["primary"], color=INK, lw=0.7, ls="-" if t == BLOCK_BSA else ":")
    ax = axes[3, 1]
    pn = c.groupby(["well", "positive"]).Area.median().unstack()
    npos = c.groupby("well").positive.sum()
    pn = pn[npos.reindex(pn.index).fillna(0) >= 20]
    meta = ctx.wells.set_index("well")
    for i, (w, r) in enumerate(pn.iterrows()):
        m = meta.loc[w]
        ax.plot([i, i], [r.get(False), r.get(True)], color=REAGENT_COLOR[m.reagent], lw=0.8)
        ax.scatter(i, r.get(False), s=16, marker=BLOCK_MARKER[m.block], facecolor="white",
                   edgecolor=REAGENT_COLOR[m.reagent], lw=0.9, zorder=3)
        ax.scatter(i, r.get(True), s=20, marker=BLOCK_MARKER[m.block], color=well_color(m.reagent, m.dose),
                   edgecolor=REAGENT_COLOR[m.reagent], lw=0.9, zorder=3)
    ax.set_xticks(range(len(pn)), labels=[str(w) for w in pn.index], fontsize=5.5)
    ax.set_xlabel("Well (≥ 20 positive BF cells)")
    ax.set_ylabel(f"Median BF cell {axis_label('Area')}", fontsize=6.5)
    ax.set_title("h  Positive (filled) vs. negative (open) BF cells")
    ax.grid(axis="y", color=GRID, lw=0.5)
    fig.tight_layout(h_pad=1.6)
    _dose_color_legend(fig, y=0.0)
    fluo_footer(fig, ctx, f"Exploratory. Matching on QC-filtered objects: each fluo object -> nearest BF cell within "
                          f"R = {R:.1f} px (0.5 × median BF equivalent diameter) after removing the per-well median "
                          "offset. Positive = BF cell with a bright\nfluo partner. The auto-well sanity check passes "
                          "partly by construction (the gate is the P99 of auto objects). d/e: solid = replicate 1, "
                          "dashed = replicate 2; bins with < 50 cells omitted.", y=-0.03)
    save(fig, "fig12_bf_fluo_matching", ctx)


# =====================================================================================
# Figure 14 – spatial intensity (illumination / tile seams)
# =====================================================================================
def fig_intensity_spatial(ctx: Ctx) -> None:
    fl = ctx.fl
    cx, cy = fl["Centroid X"].to_numpy(float), fl["Centroid Y"].to_numpy(float)
    ext = (0.0, float(cx.max()), 0.0, float(cy.max()))
    cm = plt.get_cmap("RdBu_r")
    fig, axes = plt.subplots(3, 12, figsize=(12, 3.9))
    used = {(w.plate_row, w.plate_col): w for w in ctx.fw.itertuples()}
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
                sp.set_color("#BBBBBB")
                sp.set_linewidth(0.5)
            g = ctx.fby_well.get(w.well)
            n = 0 if g is None else len(g)
            ax.set_title(f"{w.well}{flag2(w)}  n={_fmt_count(n)}" if n else f"{w.well}", fontsize=6,
                         fontweight="normal", pad=2, loc="center")
            if n == 0:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=NODATA, hatch="////",
                                       edgecolor=HATCH, lw=0))
                ax.text(.5, .5, "no fluo", transform=ax.transAxes, ha="center", va="center", fontsize=6, color=MUTED)
                continue
            gx = np.clip(((g["Centroid X"].to_numpy(float) - ext[0]) / (ext[1] - ext[0]) * INT_SPATIAL_BINS).astype(int),
                         0, INT_SPATIAL_BINS - 1)
            gy = np.clip(((g["Centroid Y"].to_numpy(float) - ext[2]) / (ext[3] - ext[2]) * INT_SPATIAL_BINS).astype(int),
                         0, INT_SPATIAL_BINS - 1)
            rel = g[MI_COL].to_numpy(float) / np.median(g[MI_COL])
            d = pd.DataFrame({"gx": gx, "gy": gy, "rel": rel}).groupby(["gy", "gx"]).rel.agg(["median", "size"])
            M = np.full((INT_SPATIAL_BINS, INT_SPATIAL_BINS), np.nan)
            ok = d["size"] >= 5
            for (a, b), v in d.loc[ok, "median"].items():
                M[a, b] = v
            im = ax.imshow(M, origin="upper", extent=(ext[0], ext[1], ext[3], ext[2]), cmap=cm, vmin=0.7, vmax=1.3,
                           interpolation="nearest")
            ax.set_aspect("equal")
        axes[i, 0].set_ylabel(row, rotation=0, ha="right", va="center", fontsize=8, fontweight="bold")
    for j in range(12):
        axes[0, j].text(.5, 1.32, str(j + 1), transform=axes[0, j].transAxes, ha="center", fontsize=7,
                        fontweight="bold")
    fig.subplots_adjust(wspace=0.08, hspace=0.35, right=0.92)
    if im is not None:
        cax = fig.add_axes([0.935, 0.2, 0.01, 0.6])
        fig.colorbar(im, cax=cax, extend="both", label="Median Mean Intensity in bin\n/ well median (raw values)")
    fluo_footer(fig, ctx, f"{INT_SPATIAL_BINS} × {INT_SPATIAL_BINS} bins over the common image extent (image coordinates, "
                          "y downwards); bins with < 5 objects blank. Values are RAW Mean Intensities.")
    save(fig, "fig14_fluo_intensity_spatial", ctx)
    fig_tile_pattern(ctx)


def fig_tile_pattern(ctx: Ctx) -> None:
    tm = ctx.tile_model
    fig = plt.figure(figsize=(10, 10.2 if tm["strong"] else 7.0))
    gs = fig.add_gridspec(3 if tm["strong"] else 2, 3, hspace=0.55, wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    lim = max(0.05, np.nanmax(np.abs(tm["map"] - 1)))
    im = ax.imshow(tm["map"], origin="upper", cmap="RdBu_r", vmin=1 - lim, vmax=1 + lim, extent=(0, 1, 1, 0))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("MI / well median", fontsize=6.5)
    ax.set_xlabel("Position within tile, x (fraction of step)")
    ax.set_ylabel("Position within tile, y")
    ax.set_title(f"a  Within-tile pattern (amplitude {tm['amp']:.2f})", fontsize=8.5)
    ax = fig.add_subplot(gs[0, 1:])
    xs = (np.arange(TILE_BINS) + 0.5) / TILE_BINS
    ax.plot(xs, tm["prof_u"], marker="o", ms=3, color=INK, label="along x")
    ax.plot(xs, tm["prof_v"], marker="s", ms=3, color=MUTED, ls="--", label="along y")
    ax.axhline(1, color=GRID, lw=0.8)
    for edge in (TILE_OVERLAP * TILE_SIZE_PX / tm["step"],):
        ax.axvspan(0, edge, color="#F2F2F2", zorder=0, label="overlap width (position unknown)")
    ax.set_xlabel(f"Position within tile (fraction of the {tm['step']:.1f} px step)")
    ax.set_ylabel("Median Mean Intensity / well median")
    ax.set_title("b  Within-tile profiles (all wells, QC-filtered objects)", fontsize=8.5)
    ax.legend(fontsize=6)
    ax.grid(color=GRID, lw=0.5)
    ax.set_ylim(min(0.9, np.nanmin([tm["prof_u"], tm["prof_v"]]) - 0.02),
                max(1.1, np.nanmax([tm["prof_u"], tm["prof_v"]]) + 0.02))
    # c: tile-offset test
    ax = fig.add_subplot(gs[1, 0])
    ph = tm["phase"].merge(ctx.wells[["well", "reagent", "dose", "block"]], on="well")
    for r in ph.itertuples():
        ax.plot([0, 1], [r.aligned * 100, r.shifted * 100], color=REAGENT_COLOR[r.reagent], lw=0.7, alpha=0.7)
        for x, v in ((0, r.aligned), (1, r.shifted)):
            ax.scatter(x, v * 100, s=16, marker=BLOCK_MARKER[r.block], color=well_color(r.reagent, r.dose),
                       edgecolor=REAGENT_COLOR[r.reagent], lw=0.7, zorder=3)
    ax.set_xticks([0, 1], labels=["tile-aligned\ngrid", "half-tile-\nshifted grid"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("Variance of log(Mean Intensity)\nbetween grid cells (%)", fontsize=6.5)
    ax.set_title("c  Tile-offset test (one line per well)", fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.5)
    # d: regional share per well
    ax = fig.add_subplot(gs[1, 1:])
    for r in ctx.fw.itertuples():
        v = getattr(r, "spatial_between_region_share", np.nan)
        if pd.notna(v):
            ax.bar(r.well, v * 100, width=0.75, color=well_color(r.reagent, r.dose), edgecolor=REAGENT_COLOR[r.reagent],
                   lw=0.6)
    ax.axhline(SPATIAL_TREND_MIN_SHARE * 100, color=STATUS_COLOR["high"], ls="--", lw=0.9,
               label=f"'strong trend' threshold ({SPATIAL_TREND_MIN_SHARE:.0%}, judged on auto wells)")
    ax.set_xticks(ctx.fw.well, labels=[str(w) for w in ctx.fw.well], fontsize=5.5)
    ax.set_xlabel("Well")
    ax.set_ylabel(f"Variance of log(Mean Intensity)\nbetween {tm['region']:.0f} px regions (%)", fontsize=6.5)
    ax.set_title(f"d  Regional trend per well (auto-well median {tm['auto_share']:.0%})", fontsize=8.5)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.5)
    note = (f"Tile step = {TILE_SIZE_PX} px × (1 − {TILE_OVERLAP:.0%}) = {tm['step']:.1f} px; tile origin unknown. "
            "Tile-wise offsets would make the aligned grid clearly better than the shifted one (c). All panels use RAW intensities.")
    if tm["strong"]:
        fw = ctx.fw[ctx.fw.fluo_present]
        for j, (raw, flat, lab) in enumerate([("mi_median", "mi_median_flat_sensitivity", "Median Mean Intensity (a.u.)"),
                                              ("frac_bright", "frac_bright_flat_sensitivity",
                                               "Fraction of bright fluo objects")]):
            ax = fig.add_subplot(gs[2, j])
            for r in fw.itertuples():
                x, y = getattr(r, raw), getattr(r, flat)
                ax.scatter(x, y, s=22, marker=BLOCK_MARKER[r.block], color=well_color(r.reagent, r.dose),
                           edgecolor=REAGENT_COLOR[r.reagent], lw=0.8, zorder=3)
                ax.annotate(str(r.well), (x, y), fontsize=5, xytext=(3, 2), textcoords="offset points")
            vals = fw[[raw, flat]].to_numpy(float)
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--", zorder=1)
            ax.set_xlabel(f"{lab} – RAW", fontsize=6.5)
            ax.set_ylabel(f"{lab} – FLATTENED", fontsize=6.5)
            ax.set_title(f"{'e' if j == 0 else 'f'}  Sensitivity: per well", fontsize=8.5)
            ax.grid(color=GRID, lw=0.5)
        ax = fig.add_subplot(gs[2, 2])
        fa = ctx.fl
        allv = np.concatenate([fa[MI_COL].to_numpy(float), fa["mi_flat"].to_numpy(float)])
        bins = np.geomspace(np.nanmin(allv), np.nanmax(allv) * 1.01, 60)
        cen = np.sqrt(bins[:-1] * bins[1:])
        show = [w for w in ctx.wells.well[ctx.wells.reagent == "auto"] if w in ctx.fby_well][:1]
        hib = ctx.fw[(ctx.fw.reagent == "HIB19") & ctx.fw.fluo_present].sort_values("frac_bright", ascending=False)
        show += [int(hib.well.iloc[0])] if len(hib) else []
        for w, c in zip(show, [REAGENT_COLOR["auto"], REAGENT_COLOR["HIB19"]]):
            g = ctx.fby_well[w]
            for col, ls, lab in ((MI_COL, "-", "raw"), ("mi_flat", "--", "flattened")):
                h, _ = np.histogram(g[col], bins)
                ax.plot(cen, h / h.sum() * 100, color=c, ls=ls, lw=1.1, label=f"well {w} {lab}")
        ax.set_xscale("log")
        ax.set_xlabel("Mean Intensity (a.u., log)", fontsize=6.5)
        ax.set_ylabel("Objects per bin (% of well)", fontsize=6.5)
        ax.set_title("g  Distribution: raw vs flattened", fontsize=8.5)
        ax.legend(fontsize=5.5)
        ax.grid(color=GRID, lw=0.5)
        note += ("\ne–g: SENSITIVITY ANALYSIS ONLY. Flattened = Mean Intensity / (median of its "
                 f"{tm['region']:.0f} px region / well median); the bright gate is recomputed as P{FLUO_INT_PERCENTILE:g} of "
                 "flattened auto objects. Removes within-well spatial variation (also real signal if positive cells "
                 "cluster spatially). All other figures use RAW values.")
    fluo_footer(fig, ctx, note, y=0.0)
    save(fig, "fig14b_fluo_tile_pattern", ctx)


# =====================================================================================
# Figure 13 – fluo composite
# =====================================================================================
def fig_fluo_composite(ctx: Ctx, v2: bool = False) -> None:
    k = COMPOSITE_METRIC
    mt = compute_matching(ctx)
    show_d = mt["reliable"]
    fig = plt.figure(figsize=(7.6, 11.6 if show_d else 9.0))
    gs = fig.add_gridspec(4 if show_d else 3, 1, height_ratios=[1.25, 1.2, 1.25] + ([1.2] if show_d else []),
                          hspace=0.55)
    ax = fig.add_subplot(gs[0])
    out = (lambda r: r.bf_qc_status == "high" or r.fluo_qc_status == "high")
    v2f = ctx.fw.assign(present=ctx.fw.fluo_present & ctx.fw.bf_present)
    if v2:
        norm, clipped = _pct_norm(v2f.loc[v2f.present, "excess_fold"])
        m = draw_plate(ax, v2f, "excess_fold", _fmt_fold, "", flag_fn=flag2, outline_fn=out, missing_fn=why_missing,
                       norm=norm)
        fig.colorbar(m, ax=ax, fraction=0.02, pad=0.1, extend="both" if clipped else "neither",
                     label="Excess fluorescence per BF cell,\nfold over auto (scale: P2–P98)")
        ax.set_title("Excess fluorescence per BF cell, fold over autofluorescence", pad=34)
    else:
        m = draw_plate(ax, v2f, "fluo_ratio", _fmt_ratio, "", vmin=0, flag_fn=flag2, outline_fn=out, missing_fn=why_missing)
        fig.colorbar(m, ax=ax, fraction=0.02, pad=0.1, label=f"Fluo objects per\nBF cell ({fluo_sel_label(ctx)})")
        ax.set_title("Fluo objects per BF cell", pad=34)
    panel_letter(ax, "A", x=-0.03, y=1.12)
    gb = gs[1].subgridspec(1, 2, wspace=0.08)
    axb = [fig.add_subplot(gb[0, j]) for j in range(2)]
    ranges = [draw_metric_dose(a, ctx, b, f"{k}_specific", k, ref="zero") for a, b in zip(axb, BLOCKS)]
    _shared_ylim(axb, ranges, zero_floor=False)
    for a, b in zip(axb, BLOCKS):
        a.set_title(BLOCK_SHORT[b], loc="left", fontsize=8)
        a.set_xlabel(f"Dose ({DOSE_UNIT})")
    plt.setp(axb[1].get_yticklabels(), visible=False)
    axb[0].set_ylabel(metric_label(k, ctx, "specific"), fontsize=6.5)
    panel_letter(axb[0], "B", y=1.08)
    gc = gs[2].subgridspec(1, 3, wspace=0.08)
    axc = [fig.add_subplot(gc[0, j]) for j in range(3)]
    ranges = [draw_fc_panel(a, ctx, r, k, k, label=False) for a, r in zip(axc, REAGENTS)]
    _shared_ylim(axc, ranges, zero_floor=True)
    for a, r in zip(axc, REAGENTS):
        a.set_title(REAGENT_LABEL[r], fontsize=8, loc="left")
    for a in axc[1:]:
        plt.setp(a.get_yticklabels(), visible=False)
    axc[0].set_ylabel(metric_label(k, ctx), fontsize=6.5)
    panel_letter(axc[0], "C", y=1.08)
    if show_d:
        gd = gs[3].subgridspec(1, 2, wspace=0.08)
        axd = [fig.add_subplot(gd[0, j]) for j in range(2)]
        for a, b in zip(axd, BLOCKS):
            draw_posfrac_size(a, ctx, b, mt)
            a.set_title(BLOCK_SHORT[b], loc="left", fontsize=8)
        top = max(a.get_ylim()[1] for a in axd)
        for a in axd:
            a.set_ylim(0, top)
        axd[1].set_ylabel("")
        plt.setp(axd[1].get_yticklabels(), visible=False)
        panel_letter(axd[0], "D", y=1.08)
    _dose_color_legend(fig, y=0.065 if show_d else 0.06)
    fluo_footer(fig, ctx, ("A: excess integrated fluorescence per BF cell (Σ max(Mean Intensity − MI_auto, 0) × Area) divided by the "
                           "auto-well mean of the same block. " if v2 else
                           "A: ratio of two independent segmentations (not % positive cells), * = ratio > 1. ") +
                          "B: HIB19 minus isotype well (same block, dose, replicate). C: BSA vs BSA + IVIG, wells paired "
                          "by plate column.\n" + ("D: exploratory BF-to-fluo matching; positive = BF cell with a bright "
                                                   "fluo partner." if show_d else
                                                   "Matching panel omitted: " + "; ".join(mt["reasons"])), y=0.02)
    save(fig, "fig13_v2_fluo_composite" if v2 else "fig13_fluo_composite", ctx)


def fig_fluo_composite_v2(ctx: Ctx) -> None:
    fig_fluo_composite(ctx, v2=True)


# =====================================================================================
# Figure 15 – well-edge effect on fluorescence intensity and BF coverage
# =====================================================================================
def fig_edge_effect(ctx: Ctx) -> None:
    E = ctx.edge
    prof, X = E["profiles"], E["X"]
    fw = ctx.fw
    centers = (np.arange(EDGE_BINS) + 0.5) / EDGE_BINS
    rows = [("mi_median", "Median Mean Intensity (a.u.)", "a  Fluo Mean Intensity vs. distance from the well edge"),
            ("bf_coverage", "BF area coverage\n(Σ BF Area / imaged area)", "b  BF area coverage"),
            ("bf_density_per_1e6px2", "BF objects per 10⁶ px²\n(imaged area)", "c  BF object density")]
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 9.6), sharex=True, sharey="row")
    alpha = {0.0: 1.0, 0.5: 0.45, 1.0: 0.7, 2.0: 1.0}
    for i, (col, yl, ttl) in enumerate(rows):
        for j, b in enumerate(BLOCKS):
            ax = axes[i, j]
            for r in fw[fw.block == b].itertuples():
                pw = prof[prof.well == r.well].sort_values("bin")
                y = pw[col].to_numpy(float)
                if col == "mi_median":
                    y = np.where(pw.n_fluo.to_numpy() >= EDGE_MIN_OBJ_BIN, y, np.nan)
                    if r.reagent == "auto":
                        lo = np.where(pw.n_fluo.to_numpy() >= EDGE_MIN_OBJ_BIN, pw.mi_q1, np.nan)
                        hi = np.where(pw.n_fluo.to_numpy() >= EDGE_MIN_OBJ_BIN, pw.mi_q3, np.nan)
                        ax.fill_between(centers, lo, hi, color=REAGENT_COLOR["auto"], alpha=0.12, lw=0)
                ax.plot(centers, y, color=REAGENT_COLOR[r.reagent], ls="-" if r.replicate == 1 else "--",
                        lw=1.2, alpha=alpha.get(float(r.dose), 1.0), marker="o", ms=2.2)
            if E["clear"]:
                ax.axvline(X, color=INK, lw=0.8, ls=":")
            ax.grid(color=GRID, lw=0.5)
            ax.set_xlim(0, 1)
            if j == 0:
                ax.set_ylabel(yl, fontsize=7)
            if i == 0:
                ax.set_title(f"Block: {b}", pad=14)
            ax.text(0.0, 1.02, ttl if j == 0 else "", transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="bottom")
        if col == "bf_density_per_1e6px2":
            axes[i, 0].set_yscale("log")
    for ax in axes[-1]:
        ax.set_xlabel("Distance from well edge (fraction of fitted well radius; 0 = edge)")
    h = [Line2D([], [], color=REAGENT_COLOR[r], lw=2, label=REAGENT_LABEL[r]) for r in REAGENTS]
    h += [Line2D([], [], color=MUTED, ls="-", label="replicate 1"), Line2D([], [], color=MUTED, ls="--", label="replicate 2"),
          Line2D([], [], color=MUTED, alpha=0.45, lw=2, label="0.5 µg (lighter) → 2 µg (darker)")]
    if E["clear"]:
        h.append(Line2D([], [], color=INK, ls=":", label=f"edge-zone limit X = {X:.2f} R"))
    fig.tight_layout(h_pad=1.6, w_pad=0.6)
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    sm = E["summary"]
    fluo_footer(fig, ctx,
                f"Well circle fitted per well from BF + fluo centroids (diagonal-sector boundary points, robust algebraic fit): radius "
                f"{sm.radius_px.median():.0f} px (range {sm.radius_px.min():.0f}–{sm.radius_px.max():.0f}), fit RMS "
                f"{sm.fit_rms_px.median():.0f} px (max {sm.fit_rms_px.max():.0f}). The imaged square clips the well at the axis directions; "
                f"{EDGE_BINS} bins of equal width in distance; imaged area per bin is computed from the grid inside the mosaic.\n"
                f"a: median per bin (bins with < {EDGE_MIN_OBJ_BIN} objects not drawn); green band = IQR of the auto wells. "
                "b: Σ QC BF object Area / imaged area (merged objects in confluent regions inflate neither count nor, much, area). "
                "Spearman correlations: edge_effect_summary.csv.")
    save(fig, "fig15_edge_effect_intensity", ctx)
    if E["clear"]:
        fig_edge_sensitivity(ctx)


def fig_edge_sensitivity(ctx: Ctx) -> None:
    X = ctx.edge["X"]
    common = ("SENSITIVITY ANALYSIS – raw values (fig8) stay the primary result. Linear colour scale clipped at P2–P98 of the wells "
              "shown. Wells with < %d objects in the zone are hatched." % EDGE_MIN_CENTER_OBJ)
    _intensity_plates(ctx, [
        ("mi_median", "a  Median Mean Intensity, all objects (raw)", "Median Mean Intensity (a.u.)",
         "Autofluorescence (auto-well mean): " + _auto_means(ctx, "mi_median") + " a.u.", _fmt_au, False),
        ("mi_median_center_only", f"b  Median Mean Intensity, central zone only (distance from edge ≥ {X:.2f} R)",
         "Median Mean Intensity,\ncentral zone (a.u.)",
         "Autofluorescence (auto-well mean): " + _auto_means(ctx, "mi_median_center_only") + " a.u.", _fmt_au, False, "few objects"),
        ("mi_median_radial_norm", "c  Median of Mean Intensity / auto-median of the same block and radial bin",
         "Radially normalised\nMean Intensity (auto = 1)",
         "Per radial bin, divided by the median Mean Intensity of the auto wells of the same block (≥ %d auto objects per bin)." %
         EDGE_MIN_OBJ_BIN, _fmt_2, False, "no ref."),
    ], "fig15b_edge_sensitivity_plates", f"Edge-zone width X = {X:.2f} R from the pooled auto-well radial profile (within {EDGE_TOL:.0%} "
                                       f"of the interior plateau at ≥ {EDGE_PLATEAU_FRAC:.2f} R). " + common, height=7.8)


# =====================================================================================
# main
# =====================================================================================
FIGURES = {"1": fig_plate_heatmap, "2": fig_dose_response, "3": fig_morphology, "4": fig_pca,
           "5": fig_spatial, "6": fig_qc, "7": fig_composite,
           "8": fig_fluo_plate, "9": fig_fluo_dose, "10": fig_fc_block, "11": fig_fluo_morph_qc,
           "12": fig_matching, "13": fig_fluo_composite, "14": fig_intensity_spatial,
           "9v2": fig_fluo_dose_v2, "10v2": fig_fc_block_v2, "13v2": fig_fluo_composite_v2, "15": fig_edge_effect}
FLUO_FIGURES = {"8", "9", "10", "11", "12", "13", "14", "9v2", "10v2", "13v2", "15"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-cache", action="store_true", help="ignore the cache and re-read every Excel file")
    ap.add_argument("--include-small", action="store_true",
                    help=f"plot all objects instead of QC-filtered (BF Area ≥ {AREA_MIN_PX} px², "
                         f"fluo Area ≥ {FLUO_AREA_MIN_PX} px²)")
    ap.add_argument("--figures", default=",".join(FIGURES),
                    help="comma-separated figure numbers to make (default: all 1–14 plus 9v2, 10v2, 13v2, 15; e.g. 1,2,8,10)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):      # never crash on "≥", "µ", "Δ" in a cp125x console
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    include_small = args.include_small or INCLUDE_SMALL_DEFAULT
    wanted = [f.strip() for f in args.figures.split(",") if f.strip()]
    bad = [f for f in wanted if f not in FIGURES]
    if bad:
        fail(f"unknown figure(s) {bad}; choose from {list(FIGURES)}")

    meta = build_well_meta()
    check_plate_map(meta)
    print("Loading BF data ...")
    cells, info = load_cells(args.no_cache, bf_modality())
    report_completeness(meta, info)
    fl_raw = None
    if FLUO_DIR.is_dir():
        print("Loading fluorescence data ...")
        fl_raw, finfo = load_cells(args.no_cache, fluo_modality())
        report_completeness(meta, finfo)
    else:
        warn(f"fluorescence folder not found ({FLUO_DIR}) – figures 8–14 skipped")
        wanted = [k for k in wanted if k not in FLUO_FIGURES]

    cells["is_small"] = cells["Area"] < AREA_MIN_PX     # always in pixel units
    wells = build_well_table(meta, cells)
    sel = cells if include_small else cells[~cells["is_small"]]
    sel = to_physical(sel)
    ctx = Ctx(cells_all=cells, cells=sel, wells=wells, include_small=include_small)
    ctx.by_well = {int(w): g for w, g in sel.groupby("well")}
    print(f"Plotting {ctx.objects_label}: {len(sel):,} of {len(cells):,} objects.")

    export_summary(ctx)
    if fl_raw is not None:
        prepare_fluo(ctx, meta, fl_raw)
        export_fluo_summary(ctx)
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
