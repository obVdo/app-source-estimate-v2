"""
app-source-estimate-v2: Apply inverse operator to evoked data → source estimate (STC).

Inputs : inverse_operator-inv.fif (from app-inverse-v2),
         epochs or evoked FIF (sensor data).
         If the evoked FIF contains multiple conditions (from by_event_type=True),
         one STC is saved per condition.
Outputs: source_estimate[-<condition>]-stc files,
         source_estimate_fsaverage[-<condition>]-stc (morphed).
"""

import os
import sys
import re

# Must be set before any vtk/pyvista/mne.viz import
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

# Resolve brainlife_utils — try local copy first, then parent monorepo
app_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(app_dir)
for search_path in [app_dir, parent_dir]:
    if os.path.isdir(os.path.join(search_path, 'brainlife_utils')):
        sys.path.insert(0, search_path)
        break

from brainlife_utils import (
    setup_matplotlib_backend,
    load_config,
    ensure_output_dirs,
    add_info_to_product,
    add_image_to_product,
    create_product_json,
)

setup_matplotlib_backend()
import matplotlib.pyplot as plt

import mne

def _safe_label(comment):
    """Convert evoked comment to a safe filename suffix."""
    if not comment:
        return ''
    return re.sub(r'[^A-Za-z0-9_-]', '_', comment.strip())

# == SETUP ==
ensure_output_dirs('out_dir', 'out_figs', 'out_report')
report_items = []

# == LOAD CONFIG ==
config = load_config()

# == LOAD INVERSE OPERATOR ==
inv_file = config.get('inverse') or ''
if not inv_file or not os.path.isfile(inv_file):
    add_info_to_product(
        report_items,
        f"FATAL: Inverse operator not found: '{inv_file}'. Run app-inverse-v2 first.",
        "error"
    )
    create_product_json(report_items)
    sys.exit(1)

try:
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_file, verbose=True)
    add_info_to_product(report_items, f"Inverse operator loaded from {inv_file}", "info")
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not read inverse operator: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == LOAD SENSOR DATA ==
epochs_file = config.get('epo') or config.get('epochs') or None
# Resolve evoked file — try config path first, then common alternative filenames
# in the same directory (evokeds_ave.fif, ave.fif) to handle different evoked apps.
_evoked_raw = config.get('evoked') or None
evoked_file = None
if _evoked_raw:
    _candidates = [_evoked_raw]
    _evoked_dir = os.path.dirname(_evoked_raw)
    for _alt in ('evokeds_ave.fif', 'ave.fif', 'evoked-ave.fif'):
        _candidate = os.path.join(_evoked_dir, _alt)
        if _candidate not in _candidates:
            _candidates.append(_candidate)
    for _c in _candidates:
        if os.path.isfile(_c):
            evoked_file = _c
            break

evoked_list = []
try:
    if epochs_file and os.path.isfile(epochs_file):
        epochs = mne.read_epochs(epochs_file, preload=True)
        ev = epochs.average()
        ev.comment = 'grand_average'
        evoked_list = [ev]
        add_info_to_product(
            report_items,
            f"Averaged {len(epochs)} epochs → 1 evoked ({ev.nave} averages, "
            f"{len(ev.ch_names)} channels)",
            "info"
        )
    elif evoked_file and os.path.isfile(evoked_file):
        evoked_list = mne.read_evokeds(evoked_file)
        if not isinstance(evoked_list, list):
            evoked_list = [evoked_list]
        add_info_to_product(
            report_items,
            f"Loaded {len(evoked_list)} evoked(s) from {evoked_file}",
            "info"
        )
    else:
        add_info_to_product(
            report_items,
            "FATAL: No sensor data found. Set 'epochs' or 'evoked' in config.json.",
            "error"
        )
        create_product_json(report_items)
        sys.exit(1)
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not load sensor data: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == INVERSE PARAMETERS ==
method    = config.get('method') or 'dSPM'
snr       = float(config.get('snr') or 3.0)
_pick_ori = config.get('pick_ori')
pick_ori  = None if _pick_ori in (None, '', 'None') else _pick_ori
lambda2   = 1.0 / snr ** 2

valid_methods = ('MNE', 'dSPM', 'sLORETA', 'eLORETA')
if method not in valid_methods:
    add_info_to_product(
        report_items,
        f"FATAL: method='{method}' not valid. Choose from {valid_methods}.",
        "error"
    )
    create_product_json(report_items)
    sys.exit(1)

# EEG average reference + baseline for all evokeds
for ev in evoked_list:
    if any(ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH for ch in ev.info['chs']):
        ev.set_eeg_reference(projection=True)
    ev.apply_baseline((None, 0))

# == MORPH SETUP ==
fs_path      = config.get('freesurfer') or config.get('output') or None
subjects_dir = config.get('subjects_dir') or None
subject      = config.get('subject') or None

if fs_path and os.path.isdir(fs_path):
    fs_path = os.path.abspath(fs_path)
    if not subjects_dir or not subject:
        if os.path.isdir(os.path.join(fs_path, 'mri')):
            subjects_dir = subjects_dir or os.path.dirname(fs_path)
            subject      = subject or os.path.basename(fs_path)
        else:
            _subdirs = sorted([d for d in os.listdir(fs_path)
                               if os.path.isdir(os.path.join(fs_path, d, 'mri'))])
            if _subdirs:
                subjects_dir = subjects_dir or fs_path
                subject      = subject or _subdirs[0]

morph_to_fsaverage = config.get('morph_to_fsaverage', False)
morph = None   # computed once from first STC, reused for all conditions

# == APPLY INVERSE — LOOP OVER CONDITIONS ==
stc_list = []   # (label, stc) for QC figures

for ev in evoked_list:
    label = _safe_label(ev.comment) or f'cond{len(stc_list)}'
    suffix = f'_{label}' if len(evoked_list) > 1 else ''

    try:
        stc = mne.minimum_norm.apply_inverse(
            ev, inverse_operator, lambda2,
            method=method, pick_ori=pick_ori, verbose=True
        )
        add_info_to_product(
            report_items,
            f"STC [{label}] ({method}): {stc.data.shape[0]} vertices, "
            f"{stc.data.shape[1]} time points",
            "info"
        )
    except Exception as e:
        add_info_to_product(report_items, f"FATAL: apply_inverse failed for [{label}]: {e}", "error")
        create_product_json(report_items)
        sys.exit(1)

    stc_path = os.path.join('out_dir', f'source_estimate{suffix}')
    try:
        stc.save(stc_path, overwrite=True)
        add_info_to_product(report_items, f"Saved: {stc_path}", "info")
    except Exception as e:
        add_info_to_product(report_items, f"FATAL: Could not save STC [{label}]: {e}", "error")
        create_product_json(report_items)
        sys.exit(1)

    # Morph to fsaverage
    if morph_to_fsaverage and subject and subjects_dir:
        if subject == 'fsaverage':
            stc_morphed = stc
            if len(stc_list) == 0:
                add_info_to_product(report_items, "Already in fsaverage — skipping morph.", "info")
        else:
            try:
                if morph is None:
                    if not os.path.isdir(os.path.join(subjects_dir, 'fsaverage')):
                        mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)
                    morph = mne.compute_source_morph(
                        stc, subject_from=subject, subject_to='fsaverage',
                        subjects_dir=subjects_dir, verbose=True
                    )
                stc_morphed = morph.apply(stc)
                morph_path  = os.path.join('out_dir', f'source_estimate_fsaverage{suffix}')
                stc_morphed.save(morph_path, overwrite=True)
                add_info_to_product(report_items, f"Morphed → fsaverage: {morph_path}", "info")
            except Exception as e:
                add_info_to_product(report_items, f"Could not morph [{label}]: {e}", "warning")

    stc_list.append((label, stc, ev))

# == QC FIGURES ==
import numpy as np

# Grand average STC across all conditions — used for butterfly, peak, brain plot
_, stc_ref, ev_ref = stc_list[0]
if len(stc_list) > 1:
    ga_data  = np.mean([stc.data for _, stc, _ in stc_list], axis=0)
    stc_ga   = stc_ref.copy()
    stc_ga.data = ga_data
    ev_ga    = mne.grand_average([ev for _, _, ev in stc_list], drop_bads=False)
    ev_ga.comment = 'grand_average'
else:
    stc_ga = stc_ref
    ev_ga  = ev_ref

# Evoked butterfly — grand average
try:
    fig_evoked = ev_ga.plot(show=False, spatial_colors=True)
    fig_path = os.path.join('out_figs', 'evoked_butterfly.png')
    fig_evoked.savefig(fig_path, dpi=72, bbox_inches='tight')
    plt.close(fig_evoked)
    add_image_to_product(report_items, 'Evoked Response (grand avg)', filepath=fig_path)
except Exception as e:
    add_info_to_product(report_items, f"Could not plot evoked: {e}", "warning")

# Source time course — all conditions overlaid
try:
    fig_stc, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab20.colors
    for i, (label, stc, _) in enumerate(stc_list):
        t_ms    = stc.times * 1000
        mean_tc = stc.data.mean(axis=0)
        ax.plot(t_ms, mean_tc, lw=1.5, color=colors[i % len(colors)], label=label)
    if len(stc_list) > 1:
        ax.plot(stc_ga.times * 1000, stc_ga.data.mean(axis=0),
                lw=2.5, color='black', label='grand avg')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(f'Source amplitude ({method})')
    ax.set_title(f'Source Time Course — {method}  ({len(stc_list)} condition(s))')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    fig_path = os.path.join('out_figs', 'source_time_course.png')
    fig_stc.savefig(fig_path, dpi=72, bbox_inches='tight')
    plt.close(fig_stc)
    add_image_to_product(report_items, 'Source Time Course', filepath=fig_path)
except Exception as e:
    add_info_to_product(report_items, f"Could not plot source time course: {e}", "warning")

# Peak info — grand average
try:
    for hemi in ('lh', 'rh'):
        peak_vert, peak_time = stc_ga.get_peak(hemi=hemi, tmin=0)
        add_info_to_product(
            report_items,
            f"Peak grand avg ({hemi}): vertex {peak_vert} at {peak_time * 1000:.1f} ms",
            "info"
        )
except Exception as e:
    add_info_to_product(report_items, f"Could not get peak: {e}", "warning")

# Brain surface plot — grand average STC
if subject and subjects_dir:
    try:
        import numpy as np
        from qtpy.QtWidgets import QApplication
        _qapp = QApplication.instance() or QApplication(sys.argv)

        import pyvista as pv
        pv.OFF_SCREEN = True
        mne.viz.set_3d_backend('pyvistaqt')

        from mne.viz.backends._pyvista import (
            PyVistaFigure, Plotter as PVPlotter, _PyVistaRenderer, _ALL_PLOTTERS,
        )
        import mne.viz.backends.renderer as renderer_mod

        def _patched_build(self):
            if self._plotter is None:
                store_filtered = {k: v for k, v in self.store.items()
                                  if k in ('window_size', 'shape', 'border', 'multi_samples')}
                plotter = PVPlotter(off_screen=True, **store_filtered)
                plotter.background_color = self.background_color
                self._plotter = plotter
                try:
                    _ALL_PLOTTERS[plotter._id_name] = plotter
                except AttributeError:
                    pass
            if self.plotter.iren is not None:
                self.plotter.iren.initialize()
                def safe_update(stime=1, force_redraw=True):
                    self.plotter.render()
                self.plotter.update = safe_update
            return self.plotter

        PyVistaFigure._build = _patched_build

        class _OffscreenRenderer(_PyVistaRenderer):
            _kind = 'pyvistaqt'
            def show(self):
                self.figure.plotter.show(auto_close=False)
            def __getattr__(self, name):
                if name.startswith(('_window_', '_dock_', '_enable_', '_disable_')):
                    return lambda *a, **kw: None
                raise AttributeError(name)

        renderer_mod.backend._Renderer = _OffscreenRenderer

        for _hemi in ('lh', 'rh'):
            _vert, _tmax = stc_ga.get_peak(hemi=_hemi, tmin=0)
            brain = stc_ga.plot(
                hemi=_hemi, subjects_dir=subjects_dir,
                views=['lateral', 'medial'], initial_time=_tmax,
                time_unit='s', size=(800, 400), smoothing_steps=10,
                background='white', colormap='hot', time_viewer=False,
            )
            brain.add_foci(_vert, coords_as_verts=True, hemi=_hemi,
                           color='blue', scale_factor=0.6, alpha=0.5)
            brain.add_text(0.1, 0.9,
                           f'{method} ({_hemi}) — peak at {_tmax * 1000:.0f} ms',
                           'title', font_size=10)
            _fig_path = os.path.join('out_figs', f'brain_{_hemi}.png')
            brain.save_image(_fig_path)
            try:
                brain.close()
            except Exception:
                pass
            add_image_to_product(report_items,
                                 f'Brain {_hemi} (peak at {_tmax * 1000:.0f} ms)',
                                 filepath=_fig_path)
    except Exception as e:
        add_info_to_product(report_items, f"Could not render brain surface plot: {e}", "warning")

# == SAVE REPORT ==
report = mne.Report(title='Source Estimate Report')
for fpath, title in [
    (os.path.join('out_figs', 'evoked_butterfly.png'),   'Evoked Response (grand avg)'),
    (os.path.join('out_figs', 'source_time_course.png'),  f'Source Time Course ({method})'),
]:
    if os.path.isfile(fpath):
        report.add_image(fpath, title=title)
if subject and subjects_dir:
    for label, stc, _ in stc_list:
        try:
            report.add_stc(stc, title=f'STC {label} ({method})',
                           subject=subject, subjects_dir=subjects_dir,
                           n_time_points=20, stc_plot_kwargs=dict(time_viewer=False))
        except Exception as e:
            add_info_to_product(report_items, f"Could not add STC [{label}] to report: {e}", "warning")
report.save(os.path.join('out_report', 'report.html'), overwrite=True)

add_info_to_product(
    report_items,
    f"Source estimation completed: {len(stc_list)} STC(s) saved.",
    "success"
)
create_product_json(report_items)
print("Done.")
