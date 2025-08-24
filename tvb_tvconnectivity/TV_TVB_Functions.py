import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Union, List, Dict
from tvb.simulator.lab import *
import scipy as sp
import sdeint

from .FC_Methods import (
    Corr_func, PCorr_func, Prec_func, MI_func,
    ZeroLagReg_func, pwGC_func, GGC_func, Coh_func,
    MVCoh_func, PCoh_func, DCoh_func, PDCoh_func,
    DTF_func
)

def simulate_time_varying_connectivity(
    connectivities: np.ndarray,
    slice_dur: Union[float, Sequence[float]],
    coupling_gain: float,
    dt: float,
    noise_sigma: float,
    monitor_list: tuple,
    centres: np.ndarray,
    tract_lengths: np.ndarray,
):
    """
    Run a TVB simulation under time-varying connectivity, allowing each slice to have its own duration,
    carrying over state between slices, and returning outputs for any number of monitors. This function can take
    connectivity with any number or regions and slices.

    Args:
        connectivities: 3D array (n_regions, n_regions, n_slices).
        slice_dur:      Either a single float (all slices same length) or a sequence of floats of length n_slices. Durations in ms.
        coupling_gain:  Scalar gain for Linear coupling.
        dt:             Integrator time-step in ms.
        noise_sigma:    Std. dev. for additive noise per region.
        monitor_list:   Tuple of TVB Monitor instances to record.
        centres:        (n_regions, 3) array of region coordinates.
        tract_lengths:  (n_regions, n_regions) array of tract lengths.

    Returns:
        List of tuples [(times_i, data_i), ...] for each monitor in monitor_list.
    """
    # Validate monitor_list
    if monitor_list is None or not isinstance(monitor_list, (list, tuple)):
        raise ValueError("monitor_list must be provided as a tuple of Monitor instances")
    
    # Validate connectivity
    if not isinstance(connectivities, np.ndarray) or connectivities.ndim != 3:
        raise ValueError("connectivities must be a 3D NumPy array of shape (n_regions, n_regions, n_slices)")
    
    if connectivities.shape[0] != connectivities.shape[1]:
        raise ValueError("The first two dimensions of connectivities must be equal (square per slice)")
    n_regions, _, n_slices = connectivities.shape

    # Validate centres and tract_lengths
    if centres is None or centres.shape != (n_regions, 3):
        raise ValueError(f"centres must be provided with shape ({n_regions}, 3)")
    if tract_lengths is None or tract_lengths.shape != (n_regions, n_regions):
        raise ValueError(f"tract_lengths must be provided with shape ({n_regions}, {n_regions})")
    
    # Handle slice durations
    if isinstance(slice_dur, (int, float)):
        slice_durs = [float(slice_dur)] * n_slices
    elif isinstance(slice_dur, Sequence):
        if len(slice_dur) != n_slices:
            raise ValueError(
                f"When providing a sequence, slice_dur length must match number of slices ({n_slices})"
            )
        slice_durs = [float(d) for d in slice_dur]
    else:
        raise ValueError("slice_dur must be a float or sequence of floats matching number of slices")

    # Precompute labels and noise vector
    region_labels = np.array([f"R{i+1}" for i in range(n_regions)], dtype="<U128")
    
    # Prepare storage
    n_monitors = len(monitor_list)
    times_storage = [[] for _ in range(n_monitors)]
    data_storage  = [[] for _ in range(n_monitors)]

    # Checkpoint state
    history_buf    = None
    current_step   = 0
    current_state  = None
    rng_state      = None
    monitor_stock  = [None] * n_monitors
    monitor_inner  = [None] * n_monitors
    monitor_state  = [None] * n_monitors


    # Loop over slices
    for k in range(n_slices):
        # Build connectivity
        conn = connectivity.Connectivity(
            weights       = connectivities[..., k],
            tract_lengths = tract_lengths,
            region_labels = region_labels,
            centres       = centres
        )
        
        noise_slice = noise.Additive(nsig=np.array([noise_sigma]),
                                     noise_seed=np.random.randint(0, 10000)
                                    )

        
        # Instantiate simulator
        sim = simulator.Simulator(
            model        = models.Generic2dOscillator(),
            connectivity = conn,
            coupling     = coupling.Linear(a=np.array([coupling_gain])),
            integrator   = integrators.HeunStochastic(
                dt    = dt,
                noise = noise_slice
            ),
            monitors     = monitor_list,
        )
        sim.configure()

        # Restore checkpoint state if any
        if history_buf is not None:
            sim.history.buffer = history_buf
            sim.current_step   = current_step
            sim.current_state  = current_state
            sim.integrator.noise.random_stream.set_state(rng_state)
            for i, mon in enumerate(sim.monitors):
                if monitor_stock[i] is not None:
                    setattr(mon, '_stock', monitor_stock[i].copy())
                if monitor_inner[i] is not None:
                    setattr(mon, '_interim_stock', monitor_inner[i].copy())
                if monitor_state[i] is not None:
                    setattr(mon, '_state', monitor_state[i].copy())

        # Run this slice
        sim.simulation_length = slice_durs[k]
        outputs = sim.run()

        # Collect outputs 
        for i, (t_i, d_i) in enumerate(outputs):
            times_storage[i].append(t_i)
            data_storage [i].append(d_i)

        # Update checkpoint
        history_buf    = sim.history.buffer.copy()
        current_step   = sim.current_step
        current_state  = sim.current_state.copy()
        rng_state      = sim.integrator.noise.random_stream.get_state()
        for i, mon in enumerate(sim.monitors):
            monitor_stock[i] = getattr(mon, '_stock', None)
            monitor_inner[i] = getattr(mon, '_interim_stock', None)
            monitor_state[i] = getattr(mon, '_state', None)

    # Concatenate and return results
    full_outputs = []
    for i in range(n_monitors):
        full_t = np.concatenate(times_storage[i])
        full_d = np.concatenate(data_storage[i], axis=0)
        full_outputs.append((full_t, full_d))

    return full_outputs

# Mapping measure names to functions
FC_measures = {
    'Corr':     Corr_func,
    'PCorr':    PCorr_func,
    'Prec':     Prec_func,
    'MI':       MI_func,
    '0-lagReg': ZeroLagReg_func,
    'pwGC':     pwGC_func,
    'GGC':      GGC_func,
    'Coh':      Coh_func,
    'MVCoh':    MVCoh_func,
    'PCoh':     PCoh_func,
    'DCoh':     DCoh_func,
    'PDCoh':    PDCoh_func,
    'DTF':      DTF_func,
}

def plot_FC_slices(raw_monitor, slice_durations, structural_slices=None, fc_methods=None):
    times, data = raw_monitor
    times = np.asarray(times).reshape(-1)               # <- ensure 1D
    signals = data.reshape(data.shape[0], -1)           # (T,1,N,1) -> (T,N)
    T, N = signals.shape

    if fc_methods is None:
        fc_methods = list(FC_measures.keys())
    M = len(fc_methods)

    if not isinstance(slice_durations, (list, tuple)) or len(slice_durations) < 1:
        raise ValueError("slice_durations must be a list of one or more durations")
    S = len(slice_durations)

    if structural_slices is not None:
        if structural_slices.ndim != 3 or structural_slices.shape[:2] != (N, N) or structural_slices.shape[2] != S:
            raise ValueError(f"structural_slices must be shape (N, N, {S}); got {structural_slices.shape}")

    # cumulative boundaries using your (absolute) times assumption
    boundaries = [times[0]]
    for dur in slice_durations:
        boundaries.append(boundaries[-1] + dur)

    for idx in range(S):
        t0, t1 = boundaries[idx], boundaries[idx+1]
        # include right edge on the last slice
        if idx == S - 1:
            mask = (times >= t0) & (times <= t1)
        else:
            mask = (times >= t0) & (times <  t1)

        X = signals[mask, :]
        if X.size == 0:
            raise ValueError(f"No data in slice {idx+1} ({t0}-{t1})")

        FCs = []
        for m in fc_methods:
            if m not in FC_measures:
                raise ValueError(f"Unknown FC method '{m}'")
            C = np.asarray(FC_measures[m](X))
            if C.shape != (N, N):
                raise ValueError(f"FC method '{m}' returned shape {C.shape}, expected ({N},{N})")
            C = C - np.diag(np.diag(C))  # zero diagonal for viz
            FCs.append(C)

        has_struct = structural_slices is not None
        ncols = M + int(has_struct)
        fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3), squeeze=False)
        col = 0

        if has_struct:
            S_mat = structural_slices[:, :, idx].copy()  # <- copy to avoid in-place modification
            S_mat = S_mat - np.diag(np.diag(S_mat))
            vmax = np.max(np.abs(S_mat))
            vmax = vmax if vmax > 0 else 1.0
            im0 = axes[0, col].imshow(S_mat, vmin=-vmax, vmax=vmax, cmap='seismic')
            axes[0, col].set_title(f'structural slice {idx+1}')
            axes[0, col].axis('off')
            fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)
            col += 1

        for k, C in enumerate(FCs):
            ax = axes[0, col + k]
            vmax = np.max(np.abs(C))
            vmax = vmax if vmax > 0 else 1.0
            im = ax.imshow(C, vmin=-vmax, vmax=vmax, cmap='seismic')
            ax.set_title(fc_methods[k])
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Slice {idx+1}: {t0:.1f}–{t1:.1f} ms")
        plt.tight_layout(rect=[0, 0.03, 1, 0.9])
        plt.show()


def apply_physio_to_bold(
    bold, snr, dt, baseline_variance, dt_physio=0.1,
    region_to_plot=0, do_plots=True
):
    """
    Add physiological artifacts to BOLD using your existing pipeline,
    and (optionally) plot the artifacts first, then raw vs +physio side-by-side.

    Parameters
    ----------
    bold : array (T, m) or TVB-like (T,1,m,1)
    snr : float
        Target SNR = var(signal) / var(noise).
    dt : float
        BOLD sampling interval (seconds).
    baseline_variance : float or array-like (m,)
        Assumed baseline variance(s) for scaling the artifacts.
    dt_physio : float, default 0.1
    region_to_plot : int, default 0
        Which region to plot in the diagnostics.
    do_plots : bool, default True
        If True, show: (1) artifact-only plots, then (2) raw vs +physio.

    Returns
    -------
    x_with_physio : array (T', m)
        BOLD with physio artifacts added (aligned length).
    """

    # clean up BOLD input
    x_raw = np.squeeze(np.asarray(bold))
    if x_raw.ndim == 1:
        x_raw = x_raw[:, None]
    if x_raw.ndim != 2:
        raise ValueError(f"`bold` must be 2D after squeeze; got {x_raw.shape}")
    T, m = x_raw.shape
    r = int(region_to_plot)
    if not (0 <= r < m):
        raise ValueError(f"region_to_plot {r} out of range 0..{m-1}")

    # baseline variance validation
    bv = np.asarray(baseline_variance)
    if bv.ndim == 0:
        pass
    elif bv.ndim == 1 and bv.shape[0] == m:
        pass
    else:
        raise ValueError("baseline_variance must be a scalar or a 1D array of length m.")
    if np.any(bv <= 0):
        raise ValueError("baseline_variance must be positive.")

    if dt <= 0 or dt_physio <= 0:
        raise ValueError("dt and dt_physio must be positive.")
    M = int(round(dt / dt_physio))
    if M < 1:
        raise ValueError("dt must be >= dt_physio so we can decimate to the BOLD rate.")


    # Generating RW & PPG with Stratonovich Heun
    def RW_PPG_sdeint(X, t):
        import numpy as _np
        dX = _np.zeros(4)
        x, y, x1, x2 = X
        a_res = 0.75; b_res = 1; w_res = 0.3*2*_np.pi; g_res = 0.0
        dx = (a_res - b_res*x**2 - b_res*y**2)*x - (w_res - g_res*x**2 - g_res*y**2)*y
        dy = (a_res - b_res*x**2 - b_res*y**2)*y + (w_res - g_res*x**2 - g_res*y**2)*x
        dX[0] = dx; dX[1] = dy
        mu = 0.5; p1 = -0.3; p2 = 0.3; b = 1.; a = 1.; c = 14.; k = 1.5
        y1 = _np.tanh(k*x1); y2 = _np.tanh(k*x2)
        tau_minus1 = a*dx + c
        dX[2] = (-x1 + (1+mu)*y1 - b*y2 + p1) * tau_minus1
        dX[3] = (-x2 + (1+mu)*y2 + b*y1 + p2) * tau_minus1
        return dX

    def sdeint_G(X, t):
        import numpy as _np
        return _np.diag([0.25, 0.25, 0.5, 0.5])

    t_physio = (np.arange(0 + dt_physio, (T*dt) + dt_physio, dt_physio)
                .round(int(1/dt_physio/10 + 1)))
    physio_result = sdeint.stratHeun(RW_PPG_sdeint, sdeint_G, [1, 1, 0.1, 0.08], t_physio)
    RW  = physio_result[:, 0]
    PPG = physio_result[:, 3]

    
    t_RF = np.arange(0, 60, dt_physio)

    def gamma_func(tau, sigma, t):
        import numpy as _np
        g = t**(np.sqrt(tau)/sigma) * np.exp(-t/(sigma*np.sqrt(tau)))
        return g / g.max()

    tau_1c = [3.1, 1.2];  sigma_1c = [2.5, 3.2];  tau_2c = [5.6, 3.5];  sigma_2c = [0.9, 0.5];  R_c = [-1.1, -1.5]
    tau_1r = [1.9, 3.1];  sigma_1r = [2.9, 3.1];  tau_2r = [12.5, 10.5]; sigma_2r = [0.5, 0.4];  R_r = [-2.6, -4.7]

    CRF = np.zeros([t_RF.shape[0], m]); RRF = np.zeros([t_RF.shape[0], m])
    for i in range(2):
        CRF[:, i] = gamma_func(tau_1c[i], sigma_1c[i], t_RF) + R_c[i]*gamma_func(tau_2c[i], sigma_2c[i], t_RF)
        CRF[:, i] /= np.linalg.norm(CRF[:, i])
        RRF[:, i] = gamma_func(tau_1r[i], sigma_1r[i], t_RF) + R_r[i]*gamma_func(tau_2r[i], sigma_2r[i], t_RF)
        RRF[:, i] /= np.linalg.norm(RRF[:, i])

    crf = CRF[:, 0]; a = crf - crf.mean(); b = CRF[:, 1] - CRF[:, 1].mean()
    crf_n = b - (b.dot(a.T)) / (a.dot(a.T)) * a
    crf   = crf  / np.linalg.norm(crf)
    crf_n = crf_n/ np.linalg.norm(crf_n)

    rrf = RRF[:, 0]; a = rrf - rrf.mean(); b = RRF[:, 1] - RRF[:, 1].mean()
    rrf_n = b - (b.dot(a.T)) / (a.dot(a.T)) * a
    rrf   = rrf  / np.linalg.norm(rrf)
    rrf_n = rrf_n/ np.linalg.norm(rrf_n)

    counter = 1
    basket_c = np.arange(counter, m); np.random.shuffle(basket_c)
    for i in basket_c:
        a = -2*(i-1)/(m-1) + 1
        b = -4/(m-2)**2 * (i - m/2)**2 + 1
        CRF[:, counter] = (a*crf + b*crf_n) / np.linalg.norm(a*crf + b*crf_n)
        counter += 1

    counter = 1
    basket_r = np.arange(counter, m); np.random.shuffle(basket_r)
    for i in basket_r:
        a = -2*(i-1)/(m-1) + 1
        b = -4/(m-2)**2 * (i - m/2)**2 + 1
        RRF[:, counter] = (a*rrf + b*rrf_n) / np.linalg.norm(a*rrf + b*rrf_n)
        counter += 1

    # HR/RF extraction and convolution with CRF/RRF
    PPG_peaks, _ = sp.signal.find_peaks(PPG, distance=.5/dt_physio, height=.1)
    HR_tmp = 60. / (PPG_peaks[1:] - PPG_peaks[0:-1]) / dt_physio
    HR = np.interp(t_physio, t_physio[PPG_peaks[0:-1]], HR_tmp)

    RF = RW**2

    HR_conv = np.zeros([len(HR), m]); RF_conv = np.zeros_like(HR_conv); physio = np.zeros_like(HR_conv)
    for i in range(m):
        HR_conv[:, i] = np.convolve(CRF[:, i], HR - HR.mean(), mode='full')[:1 - len(CRF)]
        mu_c = HR_conv[:, i].mean();  sigma_c = HR_conv[:, i].std()
        HR_conv[:, i] = (HR_conv[:, i] - mu_c) / sigma_c

        RF_conv[:, i] = np.convolve(RRF[:, i], RF - RF.mean(), mode='full')[:1 - len(RRF)]
        mu_r = RF_conv[:, i].mean();  sigma_r = RF_conv[:, i].std()
        RF_conv[:, i] = (RF_conv[:, i] - mu_r) / sigma_r

        physio[:, i] = HR_conv[:, i] + RF_conv[:, i]

    # plot artifacts alone
    if do_plots:
        t_phys = np.arange(len(physio)) * dt_physio
        plt.figure(figsize=(7.5, 3.6))
        plt.plot(t_phys, HR_conv[:, r], '--', alpha=0.6, label='HR confound')
        plt.plot(t_phys, RF_conv[:, r], '--', alpha=0.6, label='RF confound')
        plt.plot(t_phys, physio[:, r],  '-',  color='k', label='HR+RF (sum)')
        plt.title(f"Region {r+1} — physiological artifact (physio sampling)")
        plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.show()

    
    ampl = {}
    SNRs = [snr]
    for i in SNRs:
        ampl[i] = (bv / (np.var(physio, axis=0) * i))**0.5

    # decimate physio to BOLD rate
    physio_signal = sp.signal.decimate(physio, M, axis=0)  # (T_bold, m)

    # Align lengths
    Tmix = min(x_raw.shape[0], physio_signal.shape[0])
    x_raw         = x_raw[:Tmix, :]
    physio_signal = physio_signal[:Tmix, :]

    Data = {}
    for i in SNRs:
        Data[i] = {}
        Data[i]['raw'] = x_raw
        x_physio = x_raw + ampl[i] * physio_signal
        Data[i]['physio'] = x_physio

    x_with_physio = Data[snr]['physio']

    # if user wants to plot side-by-side raw vs +physio
    if do_plots:
        t_bold = np.arange(Tmix) * dt
        # artifact at BOLD rate (what you actually add)
        artifact_r = (ampl[snr] * physio_signal)[:, r]

        plt.figure(figsize=(7.5, 3.6))
        plt.plot(t_bold, artifact_r)
        plt.axhline(0, ls='--', lw=1, alpha=0.6)
        plt.title(f"Region {r+1} — artifact only (BOLD sampling)")
        plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.grid(True)
        plt.tight_layout(); plt.show()

        # side-by-side raw vs +physio
        raw_r  = x_raw[:Tmix, r]
        with_r = x_with_physio[:Tmix, r]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        axes[0].plot(t_bold, raw_r);  axes[0].set_title(f"Region {r+1} — Raw BOLD")
        axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Signal (a.u.)"); axes[0].grid(True)
        axes[1].plot(t_bold, with_r); axes[1].set_title(f"Region {r+1} — BOLD + physio")
        axes[1].set_xlabel("Time (s)"); axes[1].grid(True)
        ymin = min(raw_r.min(), with_r.min()); ymax = max(raw_r.max(), with_r.max())
        for ax in axes: ax.set_ylim(ymin, ymax)
        plt.tight_layout(); plt.show()

    return x_with_physio