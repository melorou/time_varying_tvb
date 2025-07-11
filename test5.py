import matplotlib.pyplot as plt
import numpy as np
from tvb.simulator.lab import *
import types


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
    Run a continuous TVB simulation under time-varying connectivity,
    carrying over state between slices to avoid restart transients.

    Args:
        connectivities: 3D array of shape (n_regions, n_regions, n_slices).
        slice_dur:      Duration of each slice in ms.
        centres:        (n_regions, 3) array of region coordinates.
        tract_lengths:  (n_regions, n_regions) array of tract lengths.
        coupling_gain:  Scalar gain for Linear coupling.
        dt:             Integrator time-step in ms.
        noise_sigma:    Std. dev. for additive noise per region.
        monitor_list:   Tuple of TVB Monitor instances.

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

    # Precompute labels and noise vector
    region_labels = np.array([f"R{i+1}" for i in range(n_regions)], dtype="<U128")
 

    bold_times, bold_data = [], []
    tavg_times, tavg_data = [], []

    # Variables to carry state forward
    history_buf = None
    current_step = 0
    current_state = None
    bold_stock = None
    bold_inner = None
    rng_state = None

    for k in range(n_slices):
        # Build slice-specific connectivity and simulator
        conn = connectivity.Connectivity(
            weights       = connectivities[..., k],
            tract_lengths = tract_lengths,
            region_labels = region_labels,
            centres       = centres
        )
        sim = simulator.Simulator(
            model        = models.Generic2dOscillator(),
            connectivity = conn,
            coupling     = coupling.Linear(a=np.array([coupling_gain])),
            integrator   = integrators.HeunStochastic(
                dt    = dt,
                noise = noise.Additive(nsig=np.array([noise_sigma]))
            ),
            monitors     = monitor_list
        )
        # Configure coupling & delays & history length
        sim.configure()

        print(conn.summary_info())
        
        sim._last_weights = None
        _orig = sim._loop_compute_node_coupling
        def _logged_compute(self, step):
            w = self.connectivity.weights
            if self._last_weights is None or not np.array_equal(w, self._last_weights):
                print(f"[Step {step}] connectivity.weights changed to:\n{w}\n")
                self._last_weights = w.copy()
            return _orig(step)
        sim._loop_compute_node_coupling = types.MethodType(_logged_compute, sim)

        # Restore previous slice's state to avoid restart
        if k > 0:
            sim.history.buffer = history_buf
            sim.current_step   = current_step
            sim.current_state  = current_state
            sim.monitors[1]._stock         = bold_stock
            sim.monitors[1]._interim_stock = bold_inner
            sim.integrator.noise.random_stream.set_state(rng_state)

        # Run this slice
        sim.simulation_length = slice_dur
        (t_tavg, d_tavg), (t_b, d_b) = sim.run()

        # Collect outputs, shifting time
        tavg_times.append(t_tavg)
        tavg_data.append(d_tavg)
        bold_times.append(t_b)
        bold_data.append(d_b)

        # Checkpoint for next slice
        history_buf    = sim.history.buffer.copy()
        current_step   = sim.current_step
        current_state  = sim.current_state.copy()
        bold_stock     = sim.monitors[1]._stock.copy()
        bold_inner     = sim.monitors[1]._interim_stock.copy()
        rng_state      = sim.integrator.noise.random_stream.get_state()

    # Concatenate slices
    full_time      = np.concatenate(bold_times)
    full_bold      = np.concatenate(bold_data, axis=0)
    full_tavg_time = np.concatenate(tavg_times)
    full_tavg_data = np.concatenate(tavg_data, axis=0)

    return full_time, full_bold, full_tavg_time, full_tavg_data




"""

# Example 1

connectivities_1 = np.stack([
    np.array([[2.3, 0.9],
              [-1.2, -0.5]],),
    np.array([[0.0, 0.0],
              [0.0, 0.0],]),
    np.array([[-4.4, 1.6],
              [0.8, -2.2]],)
], axis=2)

slide_dur_1 = 20_000.0   # ms

full_time_1, full_bold_1, full_tavg_time_1, full_tavg_data_1 = simulate_time_varying_connectivity(
    connectivities=connectivities_1,
    slice_dur=slide_dur_1,
)

plt.figure(figsize=(12, 6))

plt.plot(full_time_1, full_bold_1[:, 0, :, 0])
plt.ylabel("BOLD")
plt.title("BOLD Signal Over Time for Two Regions")
plt.xlabel('Time (ms)')


# Example 2

connectivities_2 = np.stack([
    [[ 1,  -2,   3,  -4],
     [ 5,  -6,   7,  -8],
     [ 9, -10,  11, -12],
     [13, -14,  15, -16]],

    [[-1,   2,  -3,   4],
     [-5,   6,  -7,   8],
     [-9,  10, -11,  12],
     [-13,  14, -15,  16]],

    [[ 0,   0,   0,   0],
     [ 0,   0,   0,   0],
     [ 0,   0,   0,   0],
     [ 0,   0,   0,   0]]
], axis=2)

slide_dur_2 = 30_000.0   # ms

full_time_2, full_bold_2, full_tavg_time_2, full_tavg_data_2 = simulate_time_varying_connectivity(
    connectivities=connectivities_2,
    slice_dur=slide_dur_2,
)

plt.figure(figsize=(12, 6))
plt.plot(full_time_2, full_bold_2[:, 0, :, 0])
plt.ylabel("BOLD")
plt.title("BOLD Signal Over Time for Four Regions")
plt.xlabel('Time (ms)')


# Example 3

connectivities_3 = np.stack([
    np.zeros((4, 4)),   

    np.array([[ 1, -1,  2, -2],                       
              [ 3, -3,  4, -4],
              [ 5, -5,  6, -6],
              [ 7, -7,  8, -8]]),

    np.array([[-8,  8, -7,  7],                       
              [-6,  6, -5,  5],
              [-4,  4, -3,  3],
              [-2,  2, -1,  1]])
], axis=2)

slice_dur_3 = 30_000.0   # ms

full_time_3, full_bold_3, full_tavg_time_3, full_tavg_data_3 = simulate_time_varying_connectivity(
    connectivities=connectivities_3,
    slice_dur=slice_dur_3,
)

plt.figure(figsize=(12, 6))
plt.plot(full_time_3, full_bold_3[:, 0, :, 0])
plt.ylabel("BOLD")
plt.title("BOLD Signal Over Time for Four Regions")
plt.xlabel('Time (ms)')



# Example 4

connectivities_4 = np.stack(
    [np.random.randn(4, 4) for _ in range(6)],
    axis=2
)

slice_dur_4 = 30_000.0   # ms

full_time_4, full_bold_4, full_tavg_time_4, full_tavg_data_4 = simulate_time_varying_connectivity(
    connectivities=connectivities_4,
    slice_dur=slice_dur_4,
)

plt.figure(figsize=(12, 6))
plt.plot(full_time_4, full_bold_4[:, 0, :, 0])
plt.ylabel("BOLD")
plt.title("BOLD Signal Over Time for Four Regions")
plt.xlabel('Time (ms)')

plt.figure(figsize=(12, 6))
plt.plot(full_tavg_time_4, full_tavg_data_4[:, 0, :, 0])
plt.ylabel("Temporal Average")
plt.title("Temporal Average Over Time for Four Regions")
plt.xlabel('Time (ms)')


# Example 5

connectivities_5 = np.stack(
    [np.random.uniform(-5, 5, (4, 4)) for _ in range(6)],
    axis=2
)

slice_dur_5 = 30_000.0   # ms

full_time_5, full_bold_5, full_tavg_time_5, full_tavg_data_5 = simulate_time_varying_connectivity(
    connectivities=connectivities_5,
    slice_dur=slice_dur_5
)

plt.figure(figsize=(12, 6))
plt.plot(full_tavg_time_5, full_tavg_data_5[:, 0, :, 0])
plt.ylabel("Temporal Average")
plt.title("Temporal Average Over Time for Four Regions")
plt.xlabel('Time (ms)')

plt.figure(figsize=(12, 6))
plt.plot(full_time_5, full_bold_5[:, 0, :, 0])
plt.ylabel("BOLD")
plt.title("BOLD Signal Over Time for Four Regions")
plt.xlabel('Time (ms)')

"""