## Case 1: the goal of this objective is to compare this function to the normal simulation

connectivities_constant = np.stack([
        [[ 1,  -2,   3, 6],
         [ 5,  -6,   7, 7],
         [ 9, -10,  11, -12],
         [13, -14,  15, -3]],

        [[ 1,  -2,   3, 6],
         [ 5,  -6,   7, 7],
         [ 9, -10,  11, -12],
         [13, -14,  15, -3]],
    ], axis=2)


slice_duration = 5000.0

# Simple centres and tract_lengths
n_regions = connectivities_constant.shape[0]
centres = np.zeros((n_regions, 3))
tract_lengths = np.ones((n_regions, n_regions))

# Monitor list: raw and bold
monitor_list = (
    monitors.Raw(period=1.0),
    monitors.Bold(period=1000.0),
)

# Run simulation
outputs_constant = tvf.simulate_time_varying_connectivity(
    connectivities=connectivities_constant,
    slice_dur=slice_duration,
    coupling_gain=0.5,
    dt=1.0,
    noise_sigma=0.001,
    monitor_list=monitor_list,
    centres=centres,
    tract_lengths=tract_lengths
)

t_raw_c, data_raw_c = outputs_constant[0]
print("Raw data shape:", data_raw_c.shape) 
plt.figure()
plt.plot(t_raw_c, data_raw_c[:, 0, :, 0])
plt.title('Raw time-series of region 1, var 0')
plt.xlabel('Time (ms)')
plt.show()

# Unpack and plot second monitor (bold)
t_bold_c, data_bold_c = outputs_constant[1]
print("BOLD data shape:", data_bold_c.shape)
plt.figure()
plt.plot(t_bold_c, data_bold_c[:, 0, :, 0])
plt.title('BOLD signals of all regions')
plt.xlabel('Time (ms)')
plt.show()

