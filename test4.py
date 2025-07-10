import matplotlib.pyplot as plt
import numpy as np
from tvb.simulator.lab import *
from tvb.simulator.simulator import Simulator

def simulate_time_varying_connectivity(
    connectivities: np.ndarray,
    slice_dur: float,
    centres: np.ndarray = None,
    tract_lengths: np.ndarray = None,
    coupling_gain: float = 0.0126,
    dt: float = 1.0,
    noise_sigma: float = 0.001,
    monitor_list: tuple = None
):
    """
    Run a continuous TVB simulation under time-varying connectivity.

    Args:
        connectivities: 3D array of shape (n_regions, n_regions, n_slices).
        slice_dur:      Duration of each slice in ms.
        centres:        (n_regions, 3) array of region coordinates. Defaults to zeros.
        tract_lengths:  (n_regions, n_regions) array of tract lengths. Defaults to ones.
        coupling_gain:  Scalar gain 'a' for Linear coupling.
        dt:             Integrator time-step in ms.
        noise_sigma:    Standard deviation for additive noise per region.
        monitor_list:   Tuple of TVB Monitor instances. Defaults to (TemporalAverage, Bold).

    Returns:
        full_time:      1D array of concatenated BOLD times.
        full_bold:      3D array (time, regions, modes) of BOLD signals.
        full_tavg_time: 1D array of concatenated TemporalAverage times.
        full_tavg_data: 2D array (time, regions) of temporal averages.
    """
    n_regions, _, n_slices = connectivities.shape

    # Defaults
    if centres is None:
        centres = np.zeros((n_regions, 3))
    if tract_lengths is None:
        tract_lengths = np.ones((n_regions, n_regions))
    if monitor_list is None:
        monitor_list = (
            monitors.TemporalAverage(period=dt),
            monitors.Bold(period=1000.0),
        )

    # Build Connectivity
    region_labels = np.array([f"R{i+1}" for i in range(n_regions)], dtype="<U128")
    conn = connectivity.Connectivity(
        weights=connectivities[..., 0],
        tract_lengths=tract_lengths,
        region_labels=region_labels,
        centres=centres
    )

    # Configure Simulator
    sigmas = np.full(n_regions, noise_sigma)
    sim = Simulator(
        model=models.Generic2dOscillator(),
        connectivity=conn,
        coupling=coupling.Linear(a=coupling_gain),
        integrator=integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=sigmas)),
        monitors=monitor_list
    )
    sim.configure()

    # Run slices
    bold_times, bold_data = [], []
    tavg_times, tavg_data = [], []

    for k in range(n_slices):
        sim.connectivity.weights = connectivities[..., k]
        sim.simulation_length = slice_dur
        (t_tavg, d_tavg), (t_b, d_b) = sim.run()
        # Shift and collect
        tavg_times.append(t_tavg + k * slice_dur)
        tavg_data.append(d_tavg)
        bold_times.append(t_b + k * slice_dur)
        bold_data.append(d_b)

    # Concatenate
    full_time = np.concatenate(bold_times)
    full_bold = np.concatenate(bold_data, axis=0)
    full_tavg_time = np.concatenate(tavg_times)
    full_tavg_data = np.concatenate(tavg_data, axis=0)

    return full_time, full_bold, full_tavg_time, full_tavg_data

# Example of calling the function:
# full_time, full_bold, full_tavg_time, full_tavg_data = simulate_time_varying_connectivity(
#     connectivities, slice_dur
# )
