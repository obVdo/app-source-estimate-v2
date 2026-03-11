"""
app-source-estimate-v2: Apply inverse operator to evoked data → source estimate (STC).

Inputs : inverse_operator-inv.fif (from app-inverse-v2),
         epochs or evoked FIF (sensor data).
Outputs: source_estimate-stc files, source_estimate_fsaverage-stc (morphed).
"""

import os
import sys

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
        f"FATAL: Inverse operator not found: '{inv_file}'. "
        "Run app-inverse-v2 first.",
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
evoked_file = config.get('evoked') or None

evoked = None
try:
    if epochs_file and os.path.isfile(epochs_file):
        epochs = mne.read_epochs(epochs_file, preload=True)
        evoked = epochs.average()
        add_info_to_product(
            report_items,
            f"Averaged {len(epochs)} epochs → evoked ({evoked.nave} averages, "
            f"{len(evoked.ch_names)} channels)",
            "info"
        )
    elif evoked_file and os.path.isfile(evoked_file):
        evoked_list = mne.read_evokeds(evoked_file)
        evoked = evoked_list[0] if isinstance(evoked_list, list) else evoked_list
        add_info_to_product(
            report_items,
            f"Loaded evoked: {evoked.nave} averages, {len(evoked.ch_names)} channels",
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

# == APPLY INVERSE ==
method   = config.get('method') or 'dSPM'
snr      = float(config.get('snr') or 3.0)
_pick_ori = config.get('pick_ori')
pick_ori  = None if _pick_ori in (None, '', 'None') else _pick_ori
lambda2  = 1.0 / snr ** 2

valid_methods = ('MNE', 'dSPM', 'sLORETA', 'eLORETA')
if method not in valid_methods:
    add_info_to_product(
        report_items,
        f"FATAL: method='{method}' not valid. Choose from {valid_methods}.",
        "error"
    )
    create_product_json(report_items)
    sys.exit(1)

evoked.apply_baseline((None, 0))

try:
    stc = mne.minimum_norm.apply_inverse(
        evoked, inverse_operator, lambda2,
        method=method, pick_ori=pick_ori, verbose=True
    )
    add_info_to_product(
        report_items,
        f"Source estimate ({method}): {stc.data.shape[0]} vertices, "
        f"{stc.data.shape[1]} time points",
        "info"
    )
except Exception as e:
    add_info_to_product(report_items, f"FATAL: apply_inverse failed: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == SAVE STC ==
stc_path = os.path.join('out_dir', 'source_estimate')
try:
    stc.save(stc_path, overwrite=True)
    add_info_to_product(report_items, f"Saved: {stc_path}", "info")
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not save STC: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == MORPH TO FSAVERAGE ==
# Accept 'freesurfer' path shorthand or subject + subjects_dir directly.
fs_path      = config.get('freesurfer') or config.get('output') or None
subjects_dir = config.get('subjects_dir') or None
subject      = config.get('subject') or None

if fs_path and os.path.isdir(fs_path):
    fs_path = os.path.abspath(fs_path)
    if not subjects_dir or not subject:
        if os.path.isdir(os.path.join(fs_path, 'mri')):
            subjects_dir = subjects_dir or os.path.dirname(fs_path)
            subject = subject or os.path.basename(fs_path)
        else:
            _subdirs = sorted([d for d in os.listdir(fs_path)
                               if os.path.isdir(os.path.join(fs_path, d, 'mri'))])
            if _subdirs:
                subjects_dir = subjects_dir or fs_path
                subject = subject or _subdirs[0]

morph_to_fsaverage = config.get('morph_to_fsaverage', False)
stc_morphed = None

if morph_to_fsaverage and subject and subjects_dir:
    if subject == 'fsaverage':
        add_info_to_product(
            report_items, "Already in fsaverage space — skipping morph.", "info"
        )
        stc_morphed = stc
    else:
        try:
            # Ensure fsaverage is available in subjects_dir
            if not os.path.isdir(os.path.join(subjects_dir, 'fsaverage')):
                mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)
            morph = mne.compute_source_morph(
                stc, subject_from=subject,
                subject_to='fsaverage',
                subjects_dir=subjects_dir,
                verbose=True
            )
            stc_morphed = morph.apply(stc)
            morph_path  = os.path.join('out_dir', 'source_estimate_fsaverage')
            stc_morphed.save(morph_path, overwrite=True)
            add_info_to_product(
                report_items,
                f"Morphed STC → fsaverage, saved: {morph_path}",
                "info"
            )
        except Exception as e:
            add_info_to_product(
                report_items, f"Could not morph to fsaverage: {e}", "warning"
            )
elif morph_to_fsaverage:
    add_info_to_product(
        report_items,
        "Skipping morph: no subject/subjects_dir provided.",
        "info"
    )

# == QC FIGURES ==

# Evoked butterfly plot
try:
    fig_evoked = evoked.plot(show=False, spatial_colors=True)
    fig_path = os.path.join('out_figs', 'evoked_butterfly.png')
    fig_evoked.savefig(fig_path, dpi=72, bbox_inches='tight')
    plt.close(fig_evoked)
    add_image_to_product(report_items, 'Evoked Response', filepath=fig_path)
except Exception as e:
    add_info_to_product(report_items, f"Could not plot evoked: {e}", "warning")

# Source time course (mean ± std across vertices)
try:
    fig_stc, ax = plt.subplots(figsize=(10, 4))
    mean_tc = stc.data.mean(axis=0)
    std_tc  = stc.data.std(axis=0)
    t_ms    = stc.times * 1000
    ax.plot(t_ms, mean_tc, linewidth=2, color='steelblue')
    ax.fill_between(t_ms, mean_tc - std_tc, mean_tc + std_tc,
                    alpha=0.2, color='steelblue')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(f'Source amplitude ({method})')
    ax.set_title(f'Source Time Course — {method}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join('out_figs', 'source_time_course.png')
    plt.savefig(fig_path, dpi=72, bbox_inches='tight')
    plt.close(fig_stc)
    add_image_to_product(report_items, 'Source Time Course', filepath=fig_path)
except Exception as e:
    add_info_to_product(report_items, f"Could not plot source time course: {e}", "warning")

# Peak info
try:
    for hemi in ('lh', 'rh'):
        peak_vert, peak_time = stc.get_peak(hemi=hemi, tmin=0)
        add_info_to_product(
            report_items,
            f"Peak ({hemi}): vertex {peak_vert} at {peak_time * 1000:.1f} ms",
            "info"
        )
except Exception:
    try:
        peak_vert, peak_time = stc.get_peak(tmin=0)
        add_info_to_product(
            report_items,
            f"Peak: vertex {peak_vert} at {peak_time * 1000:.1f} ms",
            "info"
        )
    except Exception as e:
        add_info_to_product(report_items, f"Could not get peak: {e}", "warning")

# Brain surface plot (offscreen via QT_QPA_PLATFORM=offscreen + monkey-patch)
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
                # Keep render_window alive for subsequent time-point renders
                self.figure.plotter.show(auto_close=False)
            def __getattr__(self, name):
                if name.startswith(('_window_', '_dock_', '_enable_', '_disable_')):
                    return lambda *a, **kw: None
                raise AttributeError(name)

        renderer_mod.backend._Renderer = _OffscreenRenderer

        for _hemi in ('lh', 'rh'):
            _vert, _tmax = stc.get_peak(hemi=_hemi, tmin=0)
            brain = stc.plot(
                hemi=_hemi,
                subjects_dir=subjects_dir,
                views=['lateral', 'medial'],
                initial_time=_tmax,
                time_unit='s',
                size=(800, 400),
                smoothing_steps=10,
                background='white',
                colormap='hot',
                time_viewer=False,
            )
            brain.add_foci(
                _vert, coords_as_verts=True, hemi=_hemi,
                color='blue', scale_factor=0.6, alpha=0.5,
            )
            brain.add_text(
                0.1, 0.9,
                f'{method} ({_hemi}) — peak at {_tmax * 1000:.0f} ms',
                'title', font_size=14,
            )
            _fig_path = os.path.join('out_figs', f'brain_{_hemi}.png')
            brain.save_image(_fig_path)
            try:
                brain.close()
            except Exception:
                pass
            add_image_to_product(
                report_items,
                f'Brain {_hemi} (peak at {_tmax * 1000:.0f} ms)',
                filepath=_fig_path,
            )
    except Exception as e:
        add_info_to_product(
            report_items, f"Could not render brain surface plot: {e}", "warning"
        )
else:
    pass

# == SAVE REPORT ==
report = mne.Report(title='Source Estimate Report')
evoked_fig = os.path.join('out_figs', 'evoked_butterfly.png')
stc_fig    = os.path.join('out_figs', 'source_time_course.png')
if os.path.isfile(evoked_fig):
    report.add_image(evoked_fig, title='Evoked Response')
if os.path.isfile(stc_fig):
    report.add_image(stc_fig, title=f'Source Time Course ({method})')
if subject and subjects_dir:
    try:
        report.add_stc(
            stc, title=f'Source Estimate ({method})',
            subject=subject, subjects_dir=subjects_dir,
            n_time_points=20,
            stc_plot_kwargs=dict(time_viewer=False),
        )
    except Exception as e:
        add_info_to_product(report_items, f"Could not add STC to report: {e}", "warning")
report.save(os.path.join('out_report', 'report.html'), overwrite=True)

add_info_to_product(report_items, "Source estimation completed successfully.", "success")
create_product_json(report_items)
print("Done.")
