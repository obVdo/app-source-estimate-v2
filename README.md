# app-source-estimate-v2

Apply a pre-computed inverse operator to evoked MEG/EEG data and produce source estimates (STCs).

Part of the Brainlife source reconstruction pipeline:

```
app-coreg-v2 ──→ app-bem-v2 ──→ app-forward-v2 ──→ app-inverse-v2 ──→ [app-source-estimate-v2]
                                                          ↑
                                             app-noise-covariance-v2
```

---

## Inputs

| Config key | Brainlife datatype | Description |
|---|---|---|
| `inverse` | `neuro/inverse` | Inverse operator FIF from app-inverse-v2 |
| `evoked` | `neuro/meg/fif` | Evoked FIF — single or multi-condition (see below) |
| `epo` | `neuro/meg/epochs` | Epochs FIF (averaged to grand mean if no evoked given) |
| `freesurfer` | `neuro/freesurfer` | FreeSurfer subject directory (required for brain plots and morph) |

Either `evoked` or `epo` must be provided. `evoked` takes priority.

### Multi-condition evoked files

If the evoked FIF contains **multiple conditions** (e.g. produced by an evoked app with `by_event_type=True`), the app loops over every condition and saves one STC per condition. Condition names are taken from the `comment` field of each evoked object (e.g. `"13"`, `"Dir-Ext"`).

---

## Outputs

| File | Description |
|---|---|
| `out_dir/source_estimate-stc.{fif,h5}` | Source estimate (single condition) |
| `out_dir/source_estimate_<label>-stc.{fif,h5}` | Per-condition STCs (multi-condition input) |
| `out_dir/source_estimate_fsaverage[-<label>]-stc` | Morphed STC(s) in fsaverage space (if `morph_to_fsaverage=true`) |
| `out_figs/evoked_butterfly.png` | Evoked butterfly plot (grand average across conditions) |
| `out_figs/source_time_course.png` | Mean source amplitude over time, all conditions overlaid |
| `out_figs/brain_lh.png` / `brain_rh.png` | Brain surface at peak time (grand average STC) |
| `out_report/report.html` | Interactive HTML report with all condition STCs |
| `product.json` | Brainlife UI metadata with figure thumbnails |

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | string | `dSPM` | Inverse method: `dSPM`, `sLORETA`, `MNE`, `eLORETA` |
| `snr` | number | `3.0` | Assumed signal-to-noise ratio. Sets regularisation λ² = 1/SNR². Use `3.0` for evoked, `1.0` for single trials |
| `pick_ori` | string | _(none)_ | Source orientation: leave empty for free orientation, `"normal"` for surface-normal only (reduces 3 → 1 dipole per vertex) |
| `morph_to_fsaverage` | boolean | `false` | Morph STCs to fsaverage space for group comparison |
| `subject` | string | _(auto)_ | FreeSurfer subject name (auto-detected from `freesurfer` directory if not set) |
| `subjects_dir` | string | _(auto)_ | FreeSurfer subjects directory (auto-detected if not set) |

### Choosing `method`

- **dSPM** — most common for evoked data; noise-normalised, dimensionless activation scores; recommended default
- **sLORETA** — zero localisation error for single dipoles; slightly smoother maps than dSPM
- **MNE** — minimum-norm in physical units (Am); depth-biased toward superficial sources
- **eLORETA** — exact LORETA; good point-spread function but slower

### Choosing `snr`

- `3.0` — standard for averaged evoked responses (many trials)
- `1.0`–`2.0` — single trials or noisy data
- Higher SNR → less regularisation → sharper but noisier maps

---

## What the app does

1. Loads the inverse operator from `app-inverse-v2`
2. Loads evoked data — single or multi-condition
3. Applies EEG average reference projection (if EEG data)
4. Applies baseline correction (−∞ to 0 s)
5. For each condition: `apply_inverse(evoked, inverse_operator, λ², method)`
6. Saves one STC file per condition
7. If `morph_to_fsaverage`: computes morph once, applies to all condition STCs
8. Generates QC figures from the grand average STC
9. Adds all condition STCs to the HTML report

---

## Notes

- **Baseline is always applied** (`(None, 0)`) — make sure your epochs cover at least 200 ms pre-stimulus
- **EEG average reference** is added automatically if EEG channels are present; this is required for numerically stable source estimates
- **Morph** is skipped automatically if `subject == 'fsaverage'` (template brain, already in fsaverage space)
- **Brain surface plots** require a FreeSurfer directory to be provided via the `freesurfer` input; they are skipped gracefully if absent
- STC files are saved with MNE's default naming convention (`-lh.stc` / `-rh.stc` for surface STCs)

---

## Container

```
docker://aunnikri642/app-freesurfer-mne-source-recon
```

MNE 1.11 · Python 3.12 · 20 GB RAM · 1 h wall time

---

## Authors

Guiomar Niso, Antonio Caulín, Maximilien Chaumon, obVdo  
Based on: https://github.com/guiomar/app-source-estimate
