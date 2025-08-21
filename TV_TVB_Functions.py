import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Union, List, Dict
from tvb.simulator.lab import *

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


from FC_Methods import (
    Corr_func, PCorr_func, Prec_func, MI_func,
    ZeroLagReg_func, pwGC_func, GGC_func, Coh_func,
    MVCoh_func, PCoh_func, DCoh_func, PDCoh_func,
    DTF_func
)

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