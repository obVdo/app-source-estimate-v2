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

def _parse_time(val):
    if val in (None, '', 'None'):
        return None
    return float(val)

crop_tmin = _parse_time(config.get('tmin'))
crop_tmax = _parse_time(config.get('tmax'))

valid_methods = ('MNE', 'dSPM', 'sLORETA', 'eLORETA')
if method not in valid_methods:
    add_info_to_product(
        report_items,
        f"FATAL: method='{method}' not valid. Choose from {valid_methods}.",
        "error"
    )
    create_product_json(report_items)
    sys.exit(1)

# EEG average reference + baseline + optional crop for all evokeds
for ev in evoked_list:
    if any(ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH for ch in ev.info['chs']):
        ev.set_eeg_reference(projection=True)
    ev.apply_baseline((None, 0))
    if crop_tmin is not None or crop_tmax is not None:
        ev.crop(tmin=crop_tmin, tmax=crop_tmax)
        add_info_to_product(report_items,
                            f"Cropped [{ev.comment}] to [{crop_tmin}, {crop_tmax}] s "
                            f"→ {len(ev.times)} time points", "info")

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

# If no FreeSurfer dir was provided, fall back to built-in fsaverage for plots/morph.
if subjects_dir is None:
    _fsa_path = str(mne.datasets.fetch_fsaverage(verbose=False))
    subjects_dir = os.path.dirname(_fsa_path)
    subject      = 'fsaverage'
    add_info_to_product(report_items,
                        "No FreeSurfer dir provided — using built-in fsaverage for plots.", "info")
else:
    add_info_to_product(report_items,
                        f"Using FreeSurfer subject '{subject}' from {subjects_dir}.", "info")

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
import matplotlib.image as mpimg

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

# Source time course — one figure per condition, LH + RH side by side
def _hemi_tc(stc, hemi):
    n_lh = len(stc.vertices[0])
    return stc.data[:n_lh].mean(axis=0) if hemi == 'lh' else stc.data[n_lh:].mean(axis=0)

tc_fig_paths = {}  # label → saved png path
try:
    for label, stc, _ in stc_list:
        fig_tc, axes_tc = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        fig_tc.suptitle(f'Source Time Course — {label}  ({method})', fontsize=10)
        for ax_tc, _hl, _htitle in zip(axes_tc, ('lh', 'rh'),
                                       ('Left Hemisphere', 'Right Hemisphere')):
            ax_tc.plot(stc.times * 1000, _hemi_tc(stc, _hl), lw=1.5, color='steelblue')
            ax_tc.axhline(0, color='k', lw=0.5)
            ax_tc.axvline(0, color='k', lw=0.5, ls='--')
            ax_tc.set_xlabel('Time (ms)')
            ax_tc.set_ylabel(f'Source amplitude ({method})')
            ax_tc.set_title(_htitle)
            ax_tc.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join('out_figs', f'source_time_course_{label}.png')
        fig_tc.savefig(fig_path, dpi=72, bbox_inches='tight')
        plt.close(fig_tc)
        tc_fig_paths[label] = fig_path
    # product.json thumbnail — use grand avg if multiple conditions, else the single one
    if len(stc_list) > 1:
        fig_ga, axes_ga = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        fig_ga.suptitle(f'Source Time Course — Grand Average  ({method})', fontsize=10)
        for ax_tc, _hl, _htitle in zip(axes_ga, ('lh', 'rh'),
                                       ('Left Hemisphere', 'Right Hemisphere')):
            ax_tc.plot(stc_ga.times * 1000, _hemi_tc(stc_ga, _hl), lw=2, color='black')
            ax_tc.axhline(0, color='k', lw=0.5)
            ax_tc.axvline(0, color='k', lw=0.5, ls='--')
            ax_tc.set_xlabel('Time (ms)')
            ax_tc.set_ylabel(f'Source amplitude ({method})')
            ax_tc.set_title(_htitle)
            ax_tc.grid(True, alpha=0.3)
        plt.tight_layout()
        ga_tc_path = os.path.join('out_figs', 'source_time_course_grandavg.png')
        fig_ga.savefig(ga_tc_path, dpi=72, bbox_inches='tight')
        plt.close(fig_ga)
        add_image_to_product(report_items, 'Source Time Course (grand avg)', filepath=ga_tc_path)
    else:
        add_image_to_product(report_items, 'Source Time Course',
                             filepath=next(iter(tc_fig_paths.values())))
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

# Brain surface filmstrip — grand average STC at multiple timepoints
if subject and subjects_dir:
    try:
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

        # 6 evenly-spaced timepoints from 0 ms to epoch end
        _t_pos  = stc_ga.times[stc_ga.times >= 0]
        _n_snap = min(6, len(_t_pos))
        _snap_t = _t_pos[np.round(np.linspace(0, len(_t_pos) - 1, _n_snap)).astype(int)]

        for _hemi in ('lh', 'rh'):
            _vert, _tpeak = stc_ga.get_peak(hemi=_hemi, tmin=0)
            brain = stc_ga.plot(
                hemi=_hemi, subjects_dir=subjects_dir,
                views='lateral', initial_time=_snap_t[0],
                time_unit='s', size=400, smoothing_steps=10,
                background='white', colormap='hot', time_viewer=False,
            )
            brain.add_foci(_vert, coords_as_verts=True, hemi=_hemi,
                           color='blue', scale_factor=0.6, alpha=0.5)

            _frame_paths  = []
            _frame_labels = []
            for _t in _snap_t:
                try:
                    brain.set_time(_t)
                except Exception:
                    pass
                _tmp = os.path.join('out_figs', f'_tmp_{_hemi}_{int(_t * 1000):04d}.png')
                brain.save_image(_tmp)
                _frame_paths.append(_tmp)
                _frame_labels.append(f'{_t * 1000:.0f} ms')
            try:
                brain.close()
            except Exception:
                pass

            _n_f = len(_frame_paths)
            fig_s, axes_s = plt.subplots(1, _n_f, figsize=(3.5 * _n_f, 3))
            if _n_f == 1:
                axes_s = [axes_s]
            for ax_s, _fp, _fl in zip(axes_s, _frame_paths, _frame_labels):
                ax_s.imshow(mpimg.imread(_fp))
                ax_s.set_title(_fl, fontsize=9)
                ax_s.axis('off')
                try:
                    os.remove(_fp)
                except Exception:
                    pass
            fig_s.suptitle(f'{method} ({_hemi.upper()}) — blue dot = peak vertex  '
                           f'[peak at {_tpeak * 1000:.0f} ms]', fontsize=10)
            plt.tight_layout()
            strip_path = os.path.join('out_figs', f'brain_{_hemi}.png')
            fig_s.savefig(strip_path, dpi=120, bbox_inches='tight')
            plt.close(fig_s)
            add_image_to_product(report_items,
                                 f'Brain {_hemi.upper()} filmstrip',
                                 filepath=strip_path)
    except Exception as e:
        add_info_to_product(report_items, f"Could not render brain surface plot: {e}", "warning")

# == SAVE REPORT ==
report = mne.Report(title='Source Estimate Report')

_butterfly = os.path.join('out_figs', 'evoked_butterfly.png')
if os.path.isfile(_butterfly):
    report.add_image(_butterfly, title='Evoked Response (grand avg)')

for label, stc, _ in stc_list:
    tc_path = tc_fig_paths.get(label)
    if tc_path and os.path.isfile(tc_path):
        report.add_image(tc_path, title=f'Source Time Course — {label}')
    if subject and subjects_dir:
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
