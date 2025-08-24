# TVB Time-Varying Connectivity & Physio Add-Ons

Utilities for running **time-varying connectivity** simulations in The Virtual Brain (TVB), visualizing **per-slice functional connectivity (FC)**, and **adding physiological artifacts** (cardiac/respiratory) to simulated BOLD—while preserving continuity across slice boundaries.

* Time-varying connectivity with **checkpoint & restore** 
* Per-slice FC computation/plots for validation and analysis
* Physiological artifact mixer with **user-controlled SNR**
* Defensive validation, clear errors, and docstrings to keep things accessible

---

## What’s Included

* `simulate_time_varying_connectivity(...)`
  Run TVB with slice-wise structural connectivity (SC) while restoring history, RNG, and monitor state between slices.

* `plot_FC_slices(...)`
  Compute and plot FC matrices per slice from a high-rate Raw monitor; optionally display the structural slice alongside.

* `apply_physio_to_bold(...)`
  Generate respiration/cardiac confounds, convolve with CRF/RRF, decimate to BOLD rate, and mix at a target SNR. Plot BOLD data with added physiological artifacts.

---

## Requirements

* Python 3.9+
* [`tvb-library`](https://pypi.org/project/tvb-library/)
* `numpy`, `scipy`, `matplotlib`, `sdeint`
* A local module `FC_Methods` exporting:

  * `Corr_func, PCorr_func, Prec_func, MI_func, ZeroLagReg_func, pwGC_func, GGC_func, Coh_func, MVCoh_func, PCoh_func, DCoh_func, PDCoh_func, DTF_func`

To use functions:
* python -m venv .venv
* . .venv/Script/activate
* pip install -r requirements.txt
* import core functions from tvb_tvbconnectivy

---

## Installation

Install dependencies in a clean environment using your preferred toolchain (e.g., `pip`, `uv`, or `conda` for the Python packages listed above). Ensure `tvb-library` is available and importable before use.

---

## API Reference

### `simulate_time_varying_connectivity(connectivities, slice_dur, coupling_gain, dt, noise_sigma, monitor_list, centres, tract_lengths) -> list[(times, data)]`

Run a TVB simulation where the **long-range connectivity** changes across predefined slices, while **preserving continuity** between slices.

**Parameters**

* `connectivities`: `float (N, N, S)` — weights per slice
* `slice_dur`: `float` or `Sequence[float]` — duration(s) in **ms** (scalar for all slices or one per slice)
* `coupling_gain`: `float` — gain for `coupling.Linear(a=[…])`
* `dt`: `float` — integrator step in **ms**
* `noise_sigma`: `float` — additive noise SD (scalar accepted)
* `monitor_list`: `tuple` of TVB Monitor instances (e.g., `Raw`, `Bold`)
* `centres`: `float (N, 3)` — region coordinates
* `tract_lengths`: `float (N, N)` — tract lengths (mm) used to compute delays

**Returns**

* A list of `(times, data)` tuples in the same order as `monitor_list` (TVB’s native shapes).

**Behavior**

* For each slice: builds a fresh `Connectivity` and `Simulator`, calls `configure()` to recompute **delays and connectivity-dependent internals**, **restores continuity** (history buffer, `current_state/step`, integrator RNG, and monitors’ private buffers), runs for `slice_dur[k]`, collects outputs, and saves a new checkpoint.

---

### `plot_FC_slices(raw_monitor, slice_durations, structural_slices=None, fc_methods=None)`

Compute and display FC per slice from a Raw monitor.

**Parameters**

* `raw_monitor`: `(times, data)` as returned by `monitors.Raw`
* `slice_durations`: `list[float]` in **ms**, one per slice, used to build cumulative time masks
* `structural_slices` (optional): `float (N, N, S)` — plot alongside FC
* `fc_methods` (optional): list of names mapping to `FC_Methods` functions

**Notes**

* Input BOLD/Raw shapes from TVB are respected (e.g., `(T, 1, N, 1)`); the function reshapes internally to `(T, N)`.
* FC function outputs must be `(N, N)`; diagonals are zeroed for visualization.

---

### `apply_physio_to_bold2(bold, snr, dt, baseline_variance, dt_physio=0.1, region_to_plot=0, do_plots=True) -> np.ndarray`

Add cardiac/respiratory artifacts to BOLD with diagnostic plots (optional).

**Parameters**

* `bold`: `(T, N)` or TVB `(T, 1, N, 1)`
* `snr`: target SNR = `var(signal) / var(noise)`
* `dt`: BOLD sampling interval in **seconds**
* `baseline_variance`: scalar or `(N,)` per-region baseline variance for scaling
* `dt_physio`: physio sampling step in **seconds** (default `0.1`)
* `region_to_plot`: index for diagnostics
* `do_plots`: whether to plot artifacts and Raw vs +physio comparisons

**Returns**

* `(T', N)` BOLD with artifacts (length aligned to the shorter of BOLD/physio).

**Method (high-level)**

* Simulate RW/PPG via SDE; derive HR and RF; convolve with double-gamma CRF/RRF; standardize; decimate to BOLD rate; scale to target SNR using `baseline_variance`; add to BOLD.

---

## Conventions & Assumptions

* **Units:** TVB `dt` and monitor `period` are in **ms**; physio `dt` is in **seconds**.
* **Shapes:** Structural slices `(N, N, S)`; Raw/BOLD data follow TVB shapes and are reshaped internally when needed.
* **Monitors:** Provide a `Raw` monitor (for FC) and, optionally, a `Bold` monitor.
* **FC methods:** Each function must return `(N, N)` and not alter inputs in place.
* **Private attributes:** Restoring BOLD continuity uses private fields (`_stock`, `_interim_stock`, `_state`) that may change in future TVB versions.

---

## Troubleshooting

* **Impulses at slice starts:** Continuity wasn’t fully restored (check history and BOLD internals).
* **No change in FC across slices:** Connectivity was changed without reconfiguration; ensure each slice is configured with its own `Connectivity`.
* **Shape/length mismatches:** Verify `(N, N, S)` for connectivity and that `slice_durations` cover the full time axis.
* **Long runs with large N:** Monitor stocks and history can be memory-heavy; scale gradually.

---

## Reproducibility

Record: `tvb-library` version, `connectivities`, `slice_dur`, `dt`, `coupling_gain`, `noise_sigma`, monitor types/periods, and random seeds (if deterministic behavior is required).

---

## Version Compatibility

Developed against the current `tvb-library` (scientific core). If a future release changes monitor internals, update the continuity restore step accordingly.


