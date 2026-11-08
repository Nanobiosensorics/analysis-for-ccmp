# BF + AF546 segmentation analysis – 2026-10-30 plate (HIB19 / isotype / IVIG)

`analysis.py` reads the per-well segmentation tables of two modalities, caches them, runs a
debris QC and writes all figures and summary tables to `figures_v2/` (set by `OUT_DIR` in `analysis.py`; the older `figures/` folder is left untouched):

* **BF** (brightfield cells): `BF/<N>_bf.xlsx` → figures 1–7 (unchanged from the BF-only version)
* **Fluo** (AF546 fluorescence objects): `Fluo/<N>_af546.xlsx` → figures 8–14, combined with BF

`analysis_bf_only.py` is a byte-identical backup of the BF-only version (note: it still expects the
BF files in the working folder, so it no longer finds them since they moved to `BF/`).

## Folder layout

```
analysis.py, analysis_bf_only.py, README.md, 20261030_patemap.xlsx
BF/<N>_bf.xlsx          brightfield segmentation (read-only)
Fluo/<N>_af546.xlsx     fluorescence segmentation (read-only)
cache/                  all_cells.parquet + manifest.json (BF), fluo_objects.parquet + manifest_fluo.json (Fluo)
figures_v2/             outputs of the current version (OUT_DIR); figures/ = earlier run, not overwritten
```

Excel lock files (`~$…`, created while a file is open in Excel) are skipped explicitly with a printed note.

## Requirements

Python 3.10+ with `pandas numpy matplotlib scikit-learn scipy pyarrow openpyxl`.
Optional: `umap-learn` (fig4b; skipped with a message if missing). seaborn is not needed.

## Run

```bash
python analysis.py                    # QC-filtered objects (default), incremental cache, figures 1–14
python analysis.py --include-small    # all segmented objects (outputs get suffix _allobjects)
python analysis.py --no-cache         # re-read every Excel file (both modalities)
python analysis.py --figures 8,9,13   # only selected figures (1–14)
```

Each run starts with a **data-completeness report per modality** (present / missing wells, files
added / removed / modified since the last run, per-condition table). The cache of each modality is
incremental (file name + size + modification time); nothing about which wells exist is hard-coded.
If `Fluo/` is missing, figures 8–14 are skipped with a warning.

## Configuration (top of `analysis.py`)

| Setting | Default | Meaning |
|---|---|---|
| `DATA_DIR`, `BF_DIR`, `FLUO_DIR` | `.`, `BF`, `Fluo` | plate map / BF tables / fluo tables |
| `PIXEL_SIZE_UM` | `None` | µm per pixel; `None` = everything in px |
| `AREA_MIN_PX` | 300 | BF objects with Area < this are debris |
| `HIGH_SMALL_FRAC` / `BORDERLINE_SMALL_FRAC` | 0.15 / 0.10 | BF well flag † / borderline |
| `FLUO_AREA_MIN_PX` | **100** | fluo objects with Area < this are excluded (noise floor, see below) |
| `FLUO_HIGH_SMALL_FRAC` | 0.15 | fluo well flag ‡ |
| `FLUO_INT_MIN` | `None` | fixed intensity gate (a.u.); `None` = data-driven (next two rows) |
| `FLUO_INT_PERCENTILE`, `FLUO_INT_REFERENCE` | 99, `"auto"` | gate = P99 of Mean Intensity of the auto wells of the same block; `"iso"` alternative is always computed too |
| `MATCH_RADIUS_PX` | `None` | BF↔fluo matching radius; `None` = 0.5 × median BF equivalent diameter (= 18.4 px) |
| `MATCH_PAIR_SEARCH_PX`, `MATCH_MIN_PAIRS` | 60, 50 | registration (mutual nearest neighbours) |
| `MATCH_MAX_AUTO_POS`, `MATCH_MIN_BRIGHT_RATE` | 0.05, 0.50 | sanity criteria; fig12 is skipped if they fail |
| `TILE_SIZE_PX`, `TILE_OVERLAP`, `TILE_BINS` | 1024, 0.10, 16 | stitched-tile geometry for the tile-pattern test |
| `INT_SPATIAL_BINS`, `REGION_PX`, `REGION_MIN_OBJ` | 30, `None` (= tile step 921.6 px), 20 | spatial intensity maps / regional model |
| `SPATIAL_TREND_MIN_SHARE` | 0.25 | regional share of log-intensity variance in auto wells above which the flattened sensitivity analysis is shown |
| `COMPOSITE_METRIC` | `mi_median` | metric used in fig13 B/C |
| `SEED` | 42 | all random subsampling |

### Why `FLUO_AREA_MIN_PX = 100` (not 300)
Fluo outlines are smaller than BF outlines (median fluo object ≈ 550–1100 px² vs BF ≈ 1100 px²).
The Area density of fluo objects is flat (noise floor) below ≈100 px² and rises into the cell-sized
population above it; 100 px² = equivalent diameter 11 px (< 1/3 of the median BF cell, 37 px).

## Outputs (`figures_v2/`, PNG 300 dpi + SVG)

| File | Content |
|---|---|
| `fig1` … `fig7` | BF figures, unchanged (plate map, dose response, morphology, PCA/UMAP, spatial density, QC, composite) |
| `fig8_fluo_plate_heatmap` | **intensity-based, area-normalised** plate maps: (a) median Mean Intensity, (b) integrated fluorescence per well (`ii_total`, no normalisation), (c) integrated fluorescence per BF-segmented area (`ii_per_bf_area` = `ii_total / bf_area_qc`, a.u. per px²), (d) c minus auto-well mean of the block (`ii_per_bf_area_bgcorr`), (e) c as fold over the auto-well mean (`ii_per_bf_area_fold`). Percentile-clipped (P2–P98) linear colour scales, auto level annotated |
| `fig8b_fluo_counts_supplementary` | former count-based maps (fluo objects per well / per BF cell, bright objects per BF cell) – comparison only |
| `fig8c_per_cell_count_supplementary` | normalisation by BF cell COUNT: `int_per_bf_cell` (+ background-corrected), `excess_per_bf_cell`, `excess_fold` – supplementary because the BF count is unreliable in confluent regions and these readouts are noisy |
| `fig8d_mi_delta_p90_plate` | `delta_mi_median` (median MI − auto median of the block) and `mi_p90` per well; no counts, no BF denominator |
| `fig9_v2_*`, `fig9b_v2_*`, `fig10_v2_*`, `fig10b_v2_*`, `fig13_v2_fluo_composite` | copies of fig9/10/13 with intensity metrics instead of count ratios (`mi_median`, `mi_p90`, `excess_per_bf_cell`, `excess_fold`); originals unchanged; `compare_background_metrics.csv` compares replicate agreement |
| `fig15_edge_effect_intensity` | well-edge test: Mean Intensity (median per radial bin; IQR band for auto wells), BF area coverage and BF object density vs distance from the fitted well edge, per block, replicates as separate lines |
| `fig15b_edge_sensitivity_plates` | SENSITIVITY ANALYSIS: median Mean Intensity raw vs central-zone-only vs radially normalised (drawn only if the auto wells show a clear edge gradient) |
| `fig9_fluo_dose_response` (+`_bgcorr`) | per metric vs dose: fluo ratio, bright ratio, median Mean Intensity, integrated fluorescence per BF cell; raw and background-corrected (minus auto mean, same block) |
| `fig9b_fluo_specific_signal` | HIB19 − isotype (same block, dose, replicate) |
| `fig10_fc_block_effect` (+`b_..._bgcorr`) | **key figure**: BSA vs BSA + IVIG, wells paired by plate column, all four metrics; Δ = mean(IVIG) − mean(BSA) |
| `fig11_fluo_morphology_qc` | Area / Circularity / Eccentricity violins, small-object fraction per well, noise floor |
| `fig11b_fluo_intensity_distributions` | per-dose Mean Intensity histograms + complementary ECDF (HIB19 / iso / auto), gates marked |
| `fig11c_fluo_area_vs_intensity` | Area vs Mean Intensity density, one representative well per condition (seeded) |
| `fig12_bf_fluo_matching` | exploratory BF↔fluo matching: registration offsets, match rates, fluo-positive BF cells, positive fraction by BF size, BF Area/Circularity vs matched intensity, positive vs negative BF morphology |
| `fig13_fluo_composite` | panel A–D: fluo ratio plate, specific signal, Fc-block effect, positive fraction by size |
| `fig14_fluo_intensity_spatial` | per-well map of median Mean Intensity / well median, laid out like the plate (RAW values) |
| `fig14b_fluo_tile_pattern` | within-tile pattern, tile-offset test, regional trend per well, and the **flattened sensitivity analysis** (labelled; all other figures use raw values) |
| `well_summary.csv` | BF per-well table (unchanged) |
| `well_summary_combined.csv` | BF + Fluo per well (columns below) |
| `autofluorescence_intensity.csv` | auto wells in intensity terms: `mi_median`, `ii_total`, `int_per_bf_cell` per auto well and block mean |
| `edge_effect_summary.csv` | per well: fitted well circle (centre, radius, fit RMS, flag), Spearman rho of Mean Intensity vs distance from edge (per object, and per radial bin), edge-zone vs interior median MI and BF coverage |
| `edge_effect_radial_profiles.csv` | per well and radial bin: imaged area, BF objects / area / coverage / density, fluo objects and MI median/IQR |
| `compare_background_metrics.csv` | count/all-object (v1) vs intensity (v2) metrics: replicate agreement, auto replicate agreement, fraction of negative background-corrected wells |
| `bf_fluo_matching.csv` | per-well registration offsets, match rates, fluo-positive BF fraction |
| `*_allobjects.*` | the same with `--include-small` |

### Main columns of `well_summary_combined.csv`
`n_bf_total/small/qc`, `n_fluo_total/small/qc`, `fluo_fraction_small`, `bf_qc_status`, `fluo_qc_status`,
`fluo_ratio` = n_fluo_qc / n_bf_qc (**"Fluo objects per BF cell (QC-filtered)" – not % positive**; it
is a ratio of two independent segmentations, `fluo_ratio_gt1` flags ratios > 1), `n_bright`,
`frac_bright`, `bright_ratio` (bright objects per BF cell), `mi_median/q1/q3`, `ii_median`, `ii_total`,
`int_per_bf_cell` (= total integrated intensity / n_bf_qc, "integrated fluorescence per BF cell"),
`int_threshold` (+ `_auto`, `_iso`, `_source`), `n_bright_isothr`, `bright_ratio_isothr`, and for each
metric `*_auto_mean`, `*_bgcorr` (minus auto mean, same block) and `*_specific` (HIB19 − iso, same
block/dose/replicate). Columns ending in `_flat_sensitivity` are the flattened sensitivity values;
`spatial_between_region_share` is the regional share of log-intensity variance per well.

## Intensity

* Columns are read by name: `Mean Intensity` and `Integral Intensity` (stored as "Integrated
  Intensity"). Integrated = Mean × Area holds for every object (max relative deviation 6.7·10⁻¹⁶).
* Mean Intensity ranges ≈ 7–148 a.u.; no zero/negative values, so no sign of background subtraction.
  28 objects sit exactly at the maximum 148 and the bright tail flattens towards it – possible
  clipping/saturation of the brightest objects (they are few).
* Units are arbitrary (**a.u.**). Comparisons between wells **assume identical exposure, gain and
  illumination** for all wells; this could not be checked from the tables.
* Intensity depends on the segmented outline (Mean Intensity = integrated / outline area).
* Gate: bright = Mean Intensity ≥ P99 of the auto wells of the same block (BSA 30.6, BSA + IVIG 26.4
  a.u.). The isotype-based alternative (P99 of iso wells: 50.7 / 115.9 a.u.) is in the CSV
  (`*_isothr`). Bright fractions are very sensitive to this gate (see flattened analysis).

## Spatial intensity pattern (fig14, fig14b)

* Objects lie mainly in a ring near the well edge, so maps cover an annulus.
* Within-tile pattern (1024 px tiles, 10 % overlap, step 921.6 px): weak, profiles 0.98–1.02 of the
  well median (amplitude 0.06). Tile-aligned and half-tile-shifted grids explain the same share of
  variance → **no evidence for tile-wise offsets** (tile origin is unknown, so this is a test for a
  regular grid only).
* Strong **regional** (large-scale / edge) trend: 82–87 % of the variance of log Mean Intensity in
  the auto wells lies between 922 px regions; auto wells are themselves multimodal because of it.
* Sensitivity analysis (fig14b e–g, columns `*_flat_sensitivity`): Mean Intensity divided by
  (region median / well median), gate recomputed on flattened auto objects. Per-well medians barely
  change; bright fractions change a lot (the flattened auto distribution is narrower, so its P99 drops).
  Flattening also removes real signal if positive cells cluster spatially, so it is **never** used
  in place of the raw values.

## BF↔fluo matching (fig12, exploratory)

Per-well registration from mutual nearest-neighbour pairs (median offset; offsets are small and
consistent: SD 3.0 / 2.1 px across wells), then each QC fluo object is assigned to the nearest BF cell
within R = 18.4 px; a BF cell carries the maximum Mean Intensity of its partners. Sanity: auto wells
have 0.0–0.6 % BF cells with a bright partner – this passes **partly by construction**, because the
gate is the P99 of the auto wells. With any partner (no gate) 13–47 % of auto BF cells are matched.

## Statistics and cautions

* The experimental unit is the **well** (n = 2 replicate wells per condition, 1–2 auto wells per
  block). No per-cell p-values; replicate wells are always shown individually.
* BF counts include segmentation errors (debris, clumps); the QC only removes small objects.
* Fluo counts depend on the segmentation threshold of the fluo channel: auto wells (no antibody)
  also produce many fluo objects (0.24–0.52 per BF cell).
* `fluo_ratio` is a ratio of two independent segmentations, not a fraction of positive cells.
* Tile-wise or regional illumination patterns can affect intensities (see above).
* Figure titles are neutral (no cell-type claims); observations and interpretation are kept separate.
* Missing wells are marked per modality (hatched, "no BF" / "no fluo"); ratios needing BF are blank
  for wells without BF. Currently missing: **BF 17–22, 28**; Fluo none.

## Verification done (2026-11-08)

* n per well in `well_summary_combined.csv` = data rows of the source Excel files (independent
  openpyxl count) for both modalities.
* `--no-cache` run and cached run give byte-identical outputs (CSV, PNG, SVG).
* BF outputs (`well_summary*.csv`, `morph_effect_sizes*.csv`, `pca_well_centroids*.csv`) are identical
  to those of the BF-only version.

## Known issues

* `analysis_bf_only.py` looks for `<N>_bf.xlsx` in the working folder; set its `DATA_DIR` to `BF` to use it.
* The tile origin of the stitched mosaic is unknown; the tile-pattern test assumes a regular grid.
* Possible intensity clipping at 148 a.u. is reported but not corrected.


## Why intensity is the primary metric for fig8 / autofluorescence
Object counts (`n_fluo`, `fluo_ratio`, `bright_ratio`) depend on the fluo segmentation threshold, not on signal
strength: auto wells (no antibody) also yield many objects (0.24–0.52 per BF cell). Autofluorescence is therefore
described with intensity: `mi_median` and `int_per_bf_cell` (= integrated intensity / QC BF cells; the BF count is only a
per-cell denominator). The "bright" gate stays P99 of Mean Intensity of the auto wells (same block); the same gate lines are
drawn in fig11b/fig11c. Colour scales in fig8 are percentile-clipped (P2–P98) because well values are skewed.

Note: fig9, fig10 and fig13 still contain count-based metrics (`fluo_ratio`, `bright_ratio`, incl. their `_bgcorr`
subtraction of the auto mean; fig13 A is a `fluo_ratio` plate map). They were not changed.


## New columns of `well_summary_combined.csv` (this version)
`bf_area_qc`, `fluo_area_qc` (sum of Area of QC BF / fluo objects, px²), `ii_total`, `ii_per_bf_area` (+ `_auto_mean`, `_bgcorr`, `_fold`),
`mi_p90`, `mi_auto_ref` (median Mean Intensity of all QC fluo objects in the auto wells of the block), `delta_mi_median`,
`excess_per_bf_cell`, `excess_fold` (sum of max(MI − `mi_auto_ref`, 0) × Area per BF cell, and as fold over the auto mean),
and – only when the auto wells show a clear edge gradient – the sensitivity columns `n_fluo_center_only`, `mi_median_center_only`,
`mi_p90_center_only`, `bf_area_center_only`, `ii_per_bf_area_center_only`, `n_fluo_radial_norm`, `mi_median_radial_norm`,
`mi_p90_radial_norm`. Wells without BF stay blank for the BF-area-normalised columns.

## Well-edge effect (fig15) – hypothesis, method, what the data showed
**Hypothesis (from the consultant):** cells form a brighter confluent layer near the well edge, and the BF cell count is unreliable
in confluent regions, so intensity should be normalised by cell area rather than count and the edge effect checked.

**Method.** The segmentation tables contain `Centroid X/Y` (pixels of the stitched mosaic, about 20 800 x 21 200 px). The mosaic square is
SMALLER than the circular well and clips it at the x/y axis directions; only the four diagonal sectors show the true wall. The well circle
is therefore fitted per well from boundary points (the most distant BF/fluo centroid in each of 144 angular bins, diagonal sectors only,
robust algebraic fit with MAD outlier rejection, iterated). Distance from the edge = 1 − r/R (0 = edge, 1 = centre), 20 radial bins.
Imaged area per bin is computed from a 100 px grid inside the mosaic. BF coverage = sum of QC BF Area / imaged area.

**Observations.**
* Fitted radius median 11 806 px (range 11 557–11 977, about ±1.5 %); fit RMS 17–165 px (at most 1.4 % of R); no well flagged.
* Auto wells: Spearman rho (Mean Intensity vs distance from edge) −0.72 / −0.84 / −0.89 / −0.89 (wells 25–28); median MI in the outer
  0.10 R is 18.5–21.9 a.u. versus 11.5–14.3 a.u. at >= 0.30 R (ratio about 1.5). The gradient flattens at about 0.15 R (pooled auto profile
  within 10 % of the plateau). The same negative rho is seen in almost all HIB19 and isotype wells; the exceptions are well 24 (rho about 0)
  and the sparse wells 1, 2, 5, 6, 13 (few interior objects).
* BF coverage is high near the edge (0.1–0.5 of the imaged area in the outer 0.1 R) and drops to about 0.01–0.1 beyond about 0.3–0.4 R, i.e. the
  cells are concentrated in a ring along the wall; the interior is nearly empty in most wells.
* The previously reported "strong regional trend" in the auto wells (fig14, 84 % between-region variance share) is consistent with this
  radial structure.
* Normalising by BF AREA instead of count did NOT make the auto replicates agree: `ii_per_bf_area` of auto wells 25 / 26 = 7.5 / 3.4,
  27 / 28 = 3.6 / 5.7 (a.u. per px²). The median Mean Intensity of auto wells agrees (14.6 / 16.3, 16.0 / 14.9), and after radial
  normalisation the auto wells give 0.99 / 1.04 / 1.05 / 0.96 (replicate relative difference about 5–9 %).
* Median replicate relative difference over all conditions: mi_median 0.09, mi_median_radial_norm 0.06, ii_per_bf_area 0.34,
  excess_per_bf_cell 0.37, int_per_bf_cell 0.35.

**Interpretation (not proven by these data).** An edge-concentrated, brighter cell layer would explain the position dependence of Mean
Intensity and why well-level sums depend on how many cells sit in the bright ring; integrated-intensity readouts then mostly measure
cell distribution, while median Mean Intensity (and its radially normalised version) is far more reproducible. The data cannot separate
"cells are intrinsically brighter at the edge" from "the edge is illuminated / imaged differently" (an optical effect would show the same
radial profile). Caveats: the circle fit is an estimate; the radial profile is per well and some wells have very few interior objects;
the central-zone (>= 0.15 R) readouts are blank for wells 1, 2, 13, 14 (8–35 objects) and rest on < 300 objects in wells 5, 6, 12.

**Sensitivity analysis (clearly labelled, raw values stay primary).** Zone width X = 0.15 R (first radial bin where the pooled,
plateau-normalised auto profile is within 10 % of the plateau). `*_center_only` = only objects with distance >= X (blank if < 50 objects);
`*_radial_norm` = per-object Mean Intensity divided by the median of the auto wells of the same block and radial bin (bins with
< 20 auto objects excluded), well median taken; auto about 1. Radial normalisation was implemented for Mean Intensity only (not for
integrated intensity). fig15b compares the plate maps.

## Observations reported, NOT changed (fig8–fig13)
* In the BSA block HIB19 and isotype wells are similar at 1–2 µg (median Mean Intensity HIB19 18.2 / 21.0 / 28.3 / 26.4 vs isotype
  24.1 / 24.8 / 24.2 / 27.6 a.u.): little specific signal above the isotype in this block.
* In BSA + IVIG the isotype 2 µg wells stay high (wells 23, 24: 27.9 / 30.1 a.u. vs HIB19 17.8 / 20.7); wells 21–23 (and 18, 26) carry the
  † small-object flag.
* Object area differs between blocks, which affects integrated intensity: median QC BF area about 1120 px² (BSA) vs about 940 px² (IVIG);
  median QC fluo area about 880 vs about 620 px² (pooled over wells). Earlier notes quoted ~550 vs ~1000 px²; the values above are what this
  data gives.
* The P99(auto) gate lies inside the main population of HIB19 / isotype 2 µg wells (about 1/3 of objects above the gate in wells 5, 6), so bright
  fractions are gate-sensitive (note added to fig11b/c).
* fig9, fig10 and fig13 A still contain the count-based `fluo_ratio` / `bright_ratio` and their auto-mean subtraction; the intensity-based
  copies are the `*_v2` figures.
