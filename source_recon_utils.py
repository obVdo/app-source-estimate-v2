"""
Source Reconstruction Utilities for Brainlife.io

Shared pipeline functions used by all bl-mne-* apps.
Each function is independently usable and testable.

Pipeline: input data → noise covariance → forward model → inverse operator → STC
"""

import os
import traceback
import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brainlife_utils import (
    add_info_to_product,
    add_image_to_product,
    create_product_json,
)


# ---------------------------------------------------------------------------
# Error-safe step wrapper
# ---------------------------------------------------------------------------

def safe_step(step_name, func, *args, report_items=None, **kwargs):
    """Run a pipeline step, catching errors and adding to report.

    Parameters
    ----------
    step_name : str
        Human-readable name for the step (shown in report).
    func : callable
        Pipeline function to execute.
    report_items : list
        Accumulator for report messages.

    Returns
    -------
    result : object or None
        The function's return value, or None on failure.
    ok : bool
        True if the step succeeded.
    """
    if report_items is None:
        report_items = []
    try:
        result = func(*args, **kwargs)
        add_info_to_product(report_items, f"✅ {step_name}: completed successfully", "success")
        return result, True
    except Exception as e:
        tb = traceback.format_exc()
        add_info_to_product(
            report_items,
            f"❌ {step_name} failed: {e}\n\nDetails:\n{tb}",
            "error",
        )
        return None, False


# ---------------------------------------------------------------------------
# Brainlife file path resolution
# ---------------------------------------------------------------------------

# Default filenames per Brainlife datatype.
# When Brainlife provides a directory (no File Mapping), the app must
# find the expected file inside the directory.
_DEFAULT_FILENAMES = {
    'epochs': ['meg-epo.fif', 'epochs-epo.fif'],
    'evoked': ['evoked-ave.fif', 'meg-ave.fif'],
    'raw': ['raw.fif'],
    'empty_room': ['raw.fif'],
    'forward': ['forward-fwd.fif'],
    'noise_cov': ['noise-cov.fif'],
    'inverse': ['inverse_operator-inv.fif'],
    'trans': ['trans.fif'],
    'bem': ['bem-sol.fif'],
}


def resolve_file_path(path, key=None):
    """Resolve a config path that may be a file or a Brainlife directory.

    On Brainlife, when no File Mapping is configured, the config value is
    a directory path. This function finds the actual file inside it using
    the default filename for that datatype.

    Parameters
    ----------
    path : str or None
        File path or directory path from config.json.
    key : str, optional
        Config key name (e.g., 'epochs', 'raw', 'forward') used to look up
        the default filename. If None, searches for any .fif file.

    Returns
    -------
    resolved : str or None
        Full path to the file, or None if not found.
    """
    if not path:
        return None

    # Already a file — use directly
    if os.path.isfile(path):
        return path

    # It's a directory — look for the default file inside
    if os.path.isdir(path):
        # Try known default filenames for this key
        if key and key in _DEFAULT_FILENAMES:
            for fname in _DEFAULT_FILENAMES[key]:
                candidate = os.path.join(path, fname)
                if os.path.isfile(candidate):
                    print(f"Resolved {key}: {path} -> {candidate}")
                    return candidate

        # Fallback: find any .fif file in the directory
        fif_files = [f for f in os.listdir(path) if f.endswith('.fif')]
        if len(fif_files) == 1:
            candidate = os.path.join(path, fif_files[0])
            print(f"Resolved {key or 'input'}: {path} -> {candidate}")
            return candidate
        elif len(fif_files) > 1:
            print(f"Warning: multiple .fif files in {path}: {fif_files}. "
                  f"Using first match.")
            candidate = os.path.join(path, fif_files[0])
            return candidate

        print(f"Warning: no .fif file found in directory {path}")
        return None

    # Path doesn't exist
    return path  # Return as-is; caller will handle FileNotFoundError


# ---------------------------------------------------------------------------
# Load input data
# ---------------------------------------------------------------------------

def load_input_data(config):
    """Load epochs, evoked, or raw data from config paths.

    Tries in order: 'evoked' → 'epochs' → 'raw'.
    For raw data, finds events and creates epochs automatically.

    Parameters
    ----------
    config : dict
        Configuration dictionary with file paths.

    Returns
    -------
    data : mne.Evoked | mne.Epochs
        Loaded data.

    Raises
    ------
    FileNotFoundError
        If no valid input file is found.
    """
    # Resolve paths (handles Brainlife directory inputs)
    evoked_file = resolve_file_path(config.get('evoked', ''), key='evoked')
    epochs_file = resolve_file_path(config.get('epochs', ''), key='epochs')
    raw_file = resolve_file_path(
        config.get('raw', '') or config.get('mne', ''), key='raw'
    )

    # Try evoked first
    if evoked_file and os.path.exists(evoked_file):
        evoked = mne.read_evokeds(evoked_file)
        if isinstance(evoked, list):
            evoked = evoked[0]
        print(f"Loaded evoked data from {evoked_file}: {evoked.nave} averages, "
              f"{len(evoked.ch_names)} channels")
        return evoked

    # Try epochs
    if epochs_file and os.path.exists(epochs_file):
        if epochs_file.endswith('.bdf'):
            raw = mne.io.read_raw_bdf(epochs_file, preload=True)
            events = mne.find_events(raw)
            tmin = float(config.get('tmin', -0.2))
            tmax = float(config.get('tmax', 0.5))
            epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, preload=True)
        else:
            epochs = mne.read_epochs(epochs_file, preload=True)
        print(f"Loaded {len(epochs)} epochs from {epochs_file}, "
              f"{len(epochs.ch_names)} channels")
        _check_add_eeg_ref(epochs)
        return epochs

    # Try raw
    if raw_file and os.path.exists(raw_file):
        raw = mne.io.read_raw_fif(raw_file, preload=True)
        events = mne.find_events(raw)
        tmin = float(config.get('tmin', -0.2))
        tmax = float(config.get('tmax', 0.5))
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, preload=True)
        print(f"Created {len(epochs)} epochs from raw {raw_file}")
        _check_add_eeg_ref(epochs)
        return epochs

    raise FileNotFoundError(
        "No valid input file found. Provide 'evoked', 'epochs', or 'raw' "
        f"in config.json. Checked: evoked='{evoked_file}', "
        f"epochs='{epochs_file}', raw='{raw_file}'"
    )

def _check_add_eeg_ref(data):
    """Add EEG average reference projector if missing (required for inverse)."""
    if 'eeg' in data:
        # Check if already present
        has_ref = any(proj['desc'] == 'Average EEG reference' 
                      for proj in data.info['projs'])
        if not has_ref:
            print("Adding missing EEG average reference projector")
            data.set_eeg_reference('average', projection=True)
            data.apply_proj()


# ---------------------------------------------------------------------------
# Detect modality
# ---------------------------------------------------------------------------

def detect_modality(info):
    """Detect whether data is MEG, EEG, or combined.

    Parameters
    ----------
    info : mne.Info
        MNE info object from the data.

    Returns
    -------
    modality : str
        One of 'meg', 'eeg', or 'meeg'.
    summary : dict
        Channel type counts.
    """
    ch_types = info.get_channel_types()
    meg_types = {'mag', 'grad', 'ref_meg'}
    eeg_count = sum(1 for t in ch_types if t == 'eeg')
    meg_count = sum(1 for t in ch_types if t in meg_types)

    has_eeg = eeg_count > 0
    has_meg = meg_count > 0

    summary = {'eeg_channels': eeg_count, 'meg_channels': meg_count}

    if has_meg and has_eeg:
        modality = 'meeg'
    elif has_meg:
        modality = 'meg'
    elif has_eeg:
        modality = 'eeg'
    else:
        raise ValueError(
            f"No MEG or EEG channels found. Channel types: "
            f"{set(ch_types)}"
        )

    print(f"Detected modality: {modality} "
          f"(EEG: {eeg_count}, MEG: {meg_count})")
    return modality, summary


# ---------------------------------------------------------------------------
# Noise covariance (bl-mne-noise-cov)
# ---------------------------------------------------------------------------

def compute_noise_covariance(data, tmin=None, tmax=0.0, method=None, rank=None):
    """Compute noise covariance from epochs baseline, empty-room, or ad-hoc.

    Parameters
    ----------
    data : mne.Epochs | mne.Evoked | mne.io.Raw
        Input data. Evoked → ad-hoc, Raw → full recording, Epochs → baseline.
    tmin : float or None
        Lower time bound for baseline covariance (epochs only). None = epoch start.
    tmax : float
        Upper time bound for baseline covariance (epochs only).
    method : list or None
        Estimator method(s). None defaults to ['shrunk', 'empirical'].
    rank : None | 'info' | 'full' | dict
        Rank estimation strategy. None lets MNE auto-determine rank.

    Returns
    -------
    noise_cov : mne.Covariance
        Noise covariance matrix.
    """
    if method is None:
        method = ['shrunk', 'empirical']

    # Evoked → ad-hoc
    if isinstance(data, mne.Evoked):
        print("Input is Evoked — using ad-hoc diagonal noise covariance")
        return mne.make_ad_hoc_cov(data.info)

    # Helper: try a covariance computation, falling back to 'empirical'
    # if scikit-learn is missing (MNE 1.0.2 rejects the full method list
    # if any entry requires sklearn, even if 'empirical' is also listed).
    def _try_compute(compute_fn, method, **kwargs):
        try:
            return compute_fn(method=method, **kwargs)
        except ValueError as e:
            if 'scikit-learn' in str(e) and method != ['empirical']:
                print(f"scikit-learn not available, falling back to "
                      f"method='empirical'")
                return compute_fn(method=['empirical'], **kwargs)
            raise

    # Raw (empty-room) → compute from full recording
    if isinstance(data, mne.io.BaseRaw):
        try:
            noise_cov = _try_compute(
                lambda **kw: mne.compute_raw_covariance(data, **kw),
                method, rank=rank, verbose=True
            )
            print(f"Computed noise covariance from raw recording")
            return noise_cov
        except Exception as e:
            print(f"Warning: raw covariance failed ({e}), using ad-hoc")
            return mne.make_ad_hoc_cov(data.info)

    # Epochs → compute from baseline
    try:
        cov_kwargs = dict(tmax=tmax)
        if tmin is not None:
            cov_kwargs['tmin'] = tmin
        noise_cov = _try_compute(
            lambda **kw: mne.compute_covariance(data, **cov_kwargs, **kw),
            method, rank=rank, verbose=True
        )
        tmin_str = f"{tmin}" if tmin is not None else "epoch start"
        print(f"Computed noise covariance from {len(data)} epochs "
              f"(baseline [{tmin_str}, {tmax}]s)")
        return noise_cov
    except Exception as e:
        print(f"Warning: covariance computation failed ({e}), "
              f"using ad-hoc diagonal covariance")
        return mne.make_ad_hoc_cov(data.info)


def covariance_quality_report(noise_cov, info, report_items, out_dir='out_figs'):
    """Generate quality diagnostics for a noise covariance matrix.

    Parameters
    ----------
    noise_cov : mne.Covariance
        Noise covariance matrix.
    info : mne.Info
        Data info for channel matching.
    report_items : list
        Report accumulator.
    out_dir : str
        Directory for figures.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Eigenvalue analysis
    try:
        picks = mne.pick_types(info, meg=True, eeg=True, exclude='bads')
        cov_data = noise_cov.data
        if hasattr(cov_data, 'toarray'):
            cov_data = cov_data.toarray()

        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov_data)
        eigenvalues = np.sort(eigenvalues)[::-1]  # descending

        # Condition number
        pos_eigs = eigenvalues[eigenvalues > 0]
        if len(pos_eigs) > 0:
            condition_number = pos_eigs[0] / pos_eigs[-1]
        else:
            condition_number = float('inf')

        # Effective rank (number of eigenvalues above threshold)
        threshold = eigenvalues[0] * 1e-10
        effective_rank = int(np.sum(eigenvalues > threshold))
        n_channels = len(eigenvalues)

        # Report summary
        add_info_to_product(
            report_items,
            f"📊 Covariance Quality:\n"
            f"  Channels: {n_channels}\n"
            f"  Effective rank: {effective_rank}/{n_channels}\n"
            f"  Condition number: {condition_number:.1e}\n"
            f"  Max eigenvalue: {eigenvalues[0]:.3e}\n"
            f"  Min eigenvalue: {eigenvalues[-1]:.3e}",
            "info"
        )

        # Warnings
        if condition_number > 1e12:
            add_info_to_product(
                report_items,
                f"⚠️ Very high condition number ({condition_number:.1e}). "
                f"Covariance may be poorly estimated. Consider using more data "
                f"or a regularized estimator.",
                "warning"
            )

        if effective_rank < n_channels * 0.5:
            add_info_to_product(
                report_items,
                f"⚠️ Low effective rank ({effective_rank}/{n_channels}). "
                f"Many channels may be redundant or data may be rank-deficient.",
                "warning"
            )

        # Eigenvalue spectrum plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.semilogy(range(1, len(eigenvalues) + 1), np.maximum(eigenvalues, 1e-30),
                     'o-', markersize=3, linewidth=1, color='#1976D2')
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7,
                    label=f'Rank threshold ({effective_rank})')
        ax.set_xlabel('Component')
        ax.set_ylabel('Eigenvalue')
        ax.set_title(f'Covariance Eigenvalue Spectrum (rank={effective_rank})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(out_dir, 'eigenvalue_spectrum.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        add_image_to_product(report_items, 'Eigenvalue Spectrum', filepath=fig_path)

    except Exception as e:
        add_info_to_product(
            report_items,
            f"⚠️ Could not compute covariance diagnostics: {e}",
            "warning"
        )

    # Covariance matrix visualization
    try:
        fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, info, show=False)
        fig_path = os.path.join(out_dir, 'noise_covariance.png')
        fig_cov.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig_cov)
        plt.close(fig_spectra)
        add_image_to_product(report_items, 'Noise Covariance', filepath=fig_path)
    except Exception as e:
        add_info_to_product(
            report_items,
            f"⚠️ Could not plot covariance: {e}",
            "warning"
        )


# ---------------------------------------------------------------------------
# Source space (bl-mne-forward)
# ---------------------------------------------------------------------------

def setup_source_space(subject=None, subjects_dir=None, spacing='oct6'):
    """Setup cortical source space, falling back to fsaverage if needed.

    Parameters
    ----------
    subject : str, optional
        FreeSurfer subject name.
    subjects_dir : str, optional
        FreeSurfer subjects directory.
    spacing : str
        Source space resolution (e.g., 'oct6', 'ico4').

    Returns
    -------
    src : mne.SourceSpaces
        Source space.
    subject : str
        Subject name used (may be 'fsaverage' if fallback).
    subjects_dir : str
        Subjects directory used.
    """
    # If subject MRI is available, use it
    if subject and subjects_dir and subject != 'fsaverage':
        subj_path = os.path.join(subjects_dir, subject)
        if os.path.isdir(subj_path):
            src = mne.setup_source_space(
                subject, spacing=spacing,
                subjects_dir=subjects_dir,
                add_dist=False
            )
            print(f"Created source space for {subject}: "
                  f"{sum(s['nuse'] for s in src)} sources")
            return src, subject, subjects_dir

    # Fallback to fsaverage
    print("No subject MRI found — falling back to fsaverage template")
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
    subjects_dir = os.path.dirname(fs_dir)
    subject = 'fsaverage'
    src = mne.setup_source_space(
        subject, spacing=spacing,
        subjects_dir=subjects_dir,
        add_dist=False
    )
    print(f"Created fsaverage source space: "
          f"{sum(s['nuse'] for s in src)} sources")
    return src, subject, subjects_dir


# ---------------------------------------------------------------------------
# Forward model (bl-mne-forward)
# ---------------------------------------------------------------------------

def setup_forward_model(info, trans, src, bem, modality,
                        subject=None, subjects_dir=None, mindist=5.0):
    """Compute forward solution with automatic fallback strategies.

    Strategy:
    1. If trans, src, bem all provided → compute full forward solution
    2. If fsaverage → use built-in BEM and 'fsaverage' trans
    3. EEG-only without BEM → use sphere model
    4. MEG without trans → cannot proceed

    Parameters
    ----------
    info : mne.Info
        Data info object.
    trans : str or None
        Path to -trans.fif file.
    src : mne.SourceSpaces
        Source space.
    bem : str or None
        Path to BEM solution file.
    modality : str
        'meg', 'eeg', or 'meeg'.
    subject : str, optional
        FreeSurfer subject name.
    subjects_dir : str, optional
        FreeSurfer subjects directory.
    mindist : float
        Minimum distance from source to inner skull (mm).

    Returns
    -------
    fwd : mne.Forward
        Forward solution.
    """
    use_meg = modality in ('meg', 'meeg')
    use_eeg = modality in ('eeg', 'meeg')

    # Strategy 1: Full forward with all inputs
    if trans and bem:
        if isinstance(bem, str) and os.path.exists(bem):
            bem_sol = mne.read_bem_solution(bem)
        else:
            bem_sol = bem

        if isinstance(trans, str) and os.path.exists(trans):
            trans_obj = mne.read_trans(trans)
        else:
            trans_obj = trans

        fwd = mne.make_forward_solution(
            info, trans=trans_obj, src=src, bem=bem_sol,
            meg=use_meg, eeg=use_eeg,
            mindist=mindist, n_jobs=1, verbose=True
        )
        print(f"Computed forward solution: {fwd['nsource']} sources")
        return fwd

    # Strategy 2: fsaverage with built-in BEM
    if subject == 'fsaverage' and subjects_dir:
        bem_fname = os.path.join(
            subjects_dir, 'fsaverage', 'bem',
            'fsaverage-5120-5120-5120-bem-sol.fif'
        )
        fwd = mne.make_forward_solution(
            info, trans='fsaverage', src=src,
            bem=bem_fname,
            meg=use_meg, eeg=use_eeg,
            mindist=mindist, n_jobs=1, verbose=True
        )
        print(f"Computed fsaverage forward solution: {fwd['nsource']} sources")
        return fwd

    # Strategy 3: EEG-only sphere model fallback
    if modality == 'eeg':
        print("Using sphere model for EEG-only source reconstruction")
        sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.095)
        # Use provided source space if available, otherwise create volume
        if src is None:
            vol_src = mne.setup_volume_source_space(
                subject=None, sphere=sphere, pos=10.0
            )
        else:
            vol_src = src
        fwd = mne.make_forward_solution(
            info, trans=None, src=vol_src, bem=sphere,
            meg=False, eeg=True,
            mindist=mindist, n_jobs=1, verbose=True
        )
        print(f"Computed sphere forward solution: {fwd['nsource']} sources")
        return fwd

    raise ValueError(
        f"Cannot compute forward solution for {modality} data without "
        f"trans={trans}, bem={bem}. MEG requires a head-to-MRI transform."
    )


# ---------------------------------------------------------------------------
# Make inverse operator (bl-mne-inverse)
# ---------------------------------------------------------------------------

def make_inverse_operator(data, fwd, noise_cov, loose=0.2, depth=0.8):
    """Create an inverse operator from forward solution and covariance.

    Parameters
    ----------
    data : mne.Epochs | mne.Evoked
        Input data (used for info and averaging).
    fwd : mne.Forward
        Forward solution.
    noise_cov : mne.Covariance
        Noise covariance.
    loose : float
        Loose orientation constraint (0=fixed, 1=free).
    depth : float or None
        Depth weighting exponent.

    Returns
    -------
    inverse_operator : InverseOperator
        The inverse operator.
    evoked : mne.Evoked
        The evoked data (averaged from epochs if needed).
    """
    # Get evoked
    if isinstance(data, mne.BaseEpochs):
        evoked = data.average()
        print(f"Averaged {len(data)} epochs to evoked")
    elif isinstance(data, mne.Evoked):
        evoked = data
    else:
        raise TypeError(f"Expected Epochs or Evoked, got {type(data)}")

    # Create inverse operator
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        evoked.info, fwd, noise_cov,
        loose=loose, depth=depth, verbose=True
    )
    print(f"Created inverse operator "
          f"(loose={loose}, depth={depth})")

    return inverse_operator, evoked


# ---------------------------------------------------------------------------
# Apply inverse (bl-mne-source-estimate)
# ---------------------------------------------------------------------------

def apply_inverse_to_data(evoked, inverse_operator, method='dSPM',
                          snr=3.0, pick_ori=None):
    """Apply inverse operator to evoked data to get source estimate.

    Parameters
    ----------
    evoked : mne.Evoked
        Evoked data to apply inverse to.
    inverse_operator : InverseOperator
        The inverse operator.
    method : str
        Inverse method: 'MNE', 'dSPM', 'sLORETA', or 'eLORETA'.
    snr : float
        Signal-to-noise ratio for regularization (λ² = 1/snr²).
    pick_ori : str or None
        Orientation picking strategy.

    Returns
    -------
    stc : mne.SourceEstimate
        Source time course estimate.
    """
    valid_methods = ('MNE', 'dSPM', 'sLORETA', 'eLORETA')
    if method not in valid_methods:
        raise ValueError(f"Method '{method}' not valid. Choose from {valid_methods}")

    lambda2 = 1.0 / snr ** 2
    stc = mne.minimum_norm.apply_inverse(
        evoked, inverse_operator, lambda2,
        method=method, pick_ori=pick_ori, verbose=True
    )

    print(f"Applied {method} inverse: {stc.data.shape[0]} vertices, "
          f"{stc.data.shape[1]} time points")

    return stc


def morph_to_fsaverage(stc, subject_from, subjects_dir, src_to=None):
    """Morph a source estimate to fsaverage space for group analysis.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source estimate in subject space.
    subject_from : str
        Subject name of the source estimate.
    subjects_dir : str
        FreeSurfer subjects directory.
    src_to : mne.SourceSpaces, optional
        Target source space (fsaverage).

    Returns
    -------
    stc_morphed : mne.SourceEstimate
        Source estimate in fsaverage space.
    """
    if subject_from == 'fsaverage':
        print("Already in fsaverage space, skipping morph")
        return stc

    morph = mne.compute_source_morph(
        stc, subject_from=subject_from,
        subject_to='fsaverage',
        subjects_dir=subjects_dir,
        src_to=src_to,
        verbose=True
    )
    stc_morphed = morph.apply(stc)
    print(f"Morphed STC from {subject_from} → fsaverage")
    return stc_morphed


# ---------------------------------------------------------------------------
# Backward-compat wrapper (used by full pipeline)
# ---------------------------------------------------------------------------

def make_inverse_and_apply(data, fwd, noise_cov, method='dSPM',
                           snr=3.0, loose=0.2, depth=0.8):
    """Create inverse operator and apply to evoked data (convenience wrapper).

    Combines make_inverse_operator() + apply_inverse_to_data().
    Kept for backward compatibility with tests and full pipeline.

    Returns
    -------
    stc, evoked, inverse_operator
    """
    inverse_operator, evoked = make_inverse_operator(
        data, fwd, noise_cov, loose=loose, depth=depth
    )
    stc = apply_inverse_to_data(evoked, inverse_operator,
                                 method=method, snr=snr)
    return stc, evoked, inverse_operator


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(report_items, evoked=None, noise_cov=None,
                    fwd=None, stc=None, modality=None, config=None,
                    out_dir='out_dir', subjects_dir=None, subject=None):
    """Generate comprehensive report with visualizations.

    Always writes product.json, even if some steps failed.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs('out_figs', exist_ok=True)
    os.makedirs('out_dir_report', exist_ok=True)

    # Summary header
    method = config.get('method', 'dSPM') if config else 'dSPM'
    add_info_to_product(
        report_items,
        f"📊 Source Reconstruction Summary\n"
        f"Method: {method} | Modality: {modality or 'unknown'}",
        "info"
    )

    # Evoked butterfly plot
    if evoked is not None:
        try:
            fig_evoked = evoked.plot(show=False, spatial_colors=True)
            fig_path = os.path.join('out_figs', 'evoked_butterfly.png')
            fig_evoked.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig_evoked)
            add_image_to_product(report_items, 'Evoked Response', filepath=fig_path)
        except Exception as e:
            add_info_to_product(report_items,
                                f"⚠️ Could not plot evoked: {e}", "warning")

    # Forward solution summary
    if fwd is not None:
        add_info_to_product(
            report_items,
            f"Forward solution: {fwd['nsource']} active sources, "
            f"{'fixed' if fwd['surf_ori'] else 'free'} orientation",
            "info"
        )

    # Source estimate time course
    if stc is not None:
        try:
            fig_stc = plt.figure(figsize=(10, 4))
            plt.plot(stc.times * 1000, stc.data.mean(axis=0),
                     linewidth=2, color='#2196F3')
            plt.fill_between(
                stc.times * 1000,
                stc.data.mean(axis=0) - stc.data.std(axis=0),
                stc.data.mean(axis=0) + stc.data.std(axis=0),
                alpha=0.2, color='#2196F3'
            )
            plt.xlabel('Time (ms)')
            plt.ylabel(f'Source Amplitude ({method})')
            plt.title(f'Source Time Course ({method})')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_path = os.path.join('out_figs', 'source_time_course.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig_stc)
            add_image_to_product(report_items, 'Source Time Course', filepath=fig_path)

            # Peak info
            try:
                for hemi in ('lh', 'rh'):
                    peak_vert, peak_time = stc.get_peak(hemi=hemi)
                    add_info_to_product(
                        report_items,
                        f"Peak source ({hemi}): vertex {peak_vert} "
                        f"at {peak_time * 1000:.1f} ms",
                        "info"
                    )
            except Exception:
                # Volume source estimate doesn't have hemispheres
                peak_vert, peak_time = stc.get_peak()
                add_info_to_product(
                    report_items,
                    f"Peak source: vertex {peak_vert} "
                    f"at {peak_time * 1000:.1f} ms",
                    "info"
                )

        except Exception as e:
            add_info_to_product(report_items,
                                f"⚠️ Could not plot source estimate: {e}",
                                "warning")

        # Brain surface plots (requires subjects_dir)
        _subj = subject or (config.get('subject') if config else None)
        _sdir = subjects_dir or (config.get('subjects_dir') if config else None)
        if _subj and _sdir:
            try:
                import pyvista as pv
                pv.OFF_SCREEN = True
                mne.viz.set_3d_backend('pyvistaqt')

                # Monkey-patch pyvistaqt backend for offscreen rendering
                # (avoids Qt window crash under headless/xvfb)
                from mne.viz.backends._pyvista import (
                    PyVistaFigure, Plotter as PVPlotter,
                    _PyVistaRenderer, _ALL_PLOTTERS,
                )
                import mne.viz.backends.renderer as renderer_mod

                _original_build = PyVistaFigure._build

                def _patched_build(self):
                    if self.store.get('off_screen', False):
                        if self._plotter is None:
                            store_filtered = {
                                k: v for k, v in self.store.items()
                                if k in ('window_size', 'shape', 'off_screen',
                                         'border', 'multi_samples')
                            }
                            plotter = PVPlotter(**store_filtered)
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
                    return _original_build(self)

                PyVistaFigure._build = _patched_build

                class _OffscreenRenderer(_PyVistaRenderer):
                    _kind = 'pyvistaqt'
                    def _window_initialize(self, **kwargs): pass
                    def _window_close_connect(self, func, *, after=True): pass
                    def _window_close_disconnect(self, func): pass
                    def _window_set_theme(self, theme): pass

                renderer_mod.backend._Renderer = _OffscreenRenderer

                peak_vert, peak_time = stc.get_peak()
                brain = mne.viz.Brain(
                    _subj, subjects_dir=_sdir, hemi='split',
                    views='lateral', background='white', size=(1200, 500),
                    offscreen=True, show=False,
                )
                brain.add_data(
                    stc.lh_data, colormap='hot',
                    vertices=stc.vertices[0],
                    smoothing_steps=10, time=stc.times,
                    time_label=f'{method} value',
                    hemi='lh', initial_time=peak_time,
                    colorbar=True, transparent=True,
                )
                brain.add_data(
                    stc.rh_data, colormap='hot',
                    vertices=stc.vertices[1],
                    smoothing_steps=10, time=stc.times,
                    hemi='rh', initial_time=peak_time,
                    colorbar=False, transparent=True,
                )
                brain.add_foci(
                    peak_vert, coords_as_verts=True, hemi='lh',
                    color='blue', scale_factor=0.6,
                )
                fig_path = os.path.join('out_figs', 'brain_lateral.png')
                brain.save_image(fig_path)
                try:
                    brain.close()
                except Exception:
                    pass
                add_image_to_product(
                    report_items,
                    f'Brain (lateral, peak at {peak_time*1000:.0f}ms)',
                    filepath=fig_path,
                )
                print(f"Brain surface plot saved: {fig_path}")
            except Exception as e:
                add_info_to_product(
                    report_items,
                    f"Could not render brain plots: {e}",
                    "warning",
                )

    # Write product.json
    create_product_json(report_items)
    print("Report written to product.json")


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(stc=None, inverse_operator=None, evoked=None,
                 noise_cov=None, fwd=None, src=None, out_dir='out_dir'):
    """Save all outputs to the output directory.

    Parameters
    ----------
    stc : mne.SourceEstimate, optional
    inverse_operator : InverseOperator, optional
    evoked : mne.Evoked, optional
    noise_cov : mne.Covariance, optional
    fwd : mne.Forward, optional
    src : mne.SourceSpaces, optional
    out_dir : str
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    if stc is not None:
        stc_path = os.path.join(out_dir, 'source_estimate')
        stc.save(stc_path, overwrite=True)
        saved.append(f"STC → {stc_path}")

    if inverse_operator is not None:
        inv_path = os.path.join(out_dir, 'inverse_operator-inv.fif')
        mne.minimum_norm.write_inverse_operator(inv_path, inverse_operator,
                                                 overwrite=True)
        saved.append(f"Inverse operator → {inv_path}")

    if evoked is not None:
        evo_path = os.path.join(out_dir, 'evoked-ave.fif')
        evoked.save(evo_path, overwrite=True)
        saved.append(f"Evoked → {evo_path}")

    if noise_cov is not None:
        cov_path = os.path.join(out_dir, 'noise-cov.fif')
        mne.write_cov(cov_path, noise_cov, overwrite=True)
        saved.append(f"Noise covariance → {cov_path}")

    if fwd is not None:
        fwd_path = os.path.join(out_dir, 'forward-fwd.fif')
        mne.write_forward_solution(fwd_path, fwd, overwrite=True)
        saved.append(f"Forward → {fwd_path}")

    if src is not None:
        src_path = os.path.join(out_dir, 'source_space-src.fif')
        src.save(src_path, overwrite=True)
        saved.append(f"Source space → {src_path}")

    for s in saved:
        print(f"Saved: {s}")

    return saved


# ---------------------------------------------------------------------------
# Finalize report (always called, even on failure)
# ---------------------------------------------------------------------------

def finalize(report_items, success=True):
    """Write final product.json with all accumulated report items."""
    if success:
        add_info_to_product(report_items,
                            "🎉 Completed successfully!", "success")
    else:
        add_info_to_product(report_items,
                            "⚠️ Did not complete. Review error messages above.",
                            "warning")

    create_product_json(report_items)
    print(f"Final report written (success={success})")
