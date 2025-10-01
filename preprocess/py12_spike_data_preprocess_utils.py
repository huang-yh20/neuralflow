import numpy as np

# 遍历每个trial，如果某个trial中所有神经元总放电数为0，则将这个trial剔除
def clean_spike_data_1(spike_counts, trial_count, timeepoch):
    total_rows, num_neuron = spike_counts.shape
    time_points = total_rows // trial_count
    non_empty_trials = []
    # 遍历每个trial，如果某个trial中所有神经元总放电数为0，则将这个trial剔除
    for trial_idx in range(trial_count):
        trial_data = spike_counts[trial_idx * time_points:(trial_idx + 1) * time_points, :]
        # 计算每个神经元在trial中的总放电数
        neuron_totals = np.sum(trial_data)
        # 如果所有神经元的总放电数都大于0，则保留该trial；否则，剔除
        if neuron_totals > 0:
            non_empty_trials.append(trial_idx)
        else:
            print(f"Removing trial {trial_idx} due to at least one neuron with 0 spikes")
    # 根据非空trial的索引，从原始数据中提取对应trial的数据（连续time_points行）
    filtered_trials = [
        spike_counts[trial_idx * time_points:(trial_idx + 1) * time_points, :]
        for trial_idx in non_empty_trials
    ]
    new_spike_counts = np.vstack(filtered_trials)
    
    # 如果timeepoch是列表，则只保留对应非空trial的时间段
    new_timeepoch = [timeepoch[i] for i in non_empty_trials]
    
    # 计算新的trial数量
    new_trial_count = len(non_empty_trials)
    
    return new_spike_counts, new_trial_count, new_timeepoch


# 遍历每个神经元，如果某个神经元在任意一个trial中的总放电数为0，则将这个神经元剔除
# 但实际上还是有问题，trial一多就没几个神经元了
def clean_spike_data_2(spike_counts, trial_count, timeepoch):
    total_rows, num_neuron = spike_counts.shape
    time_points = total_rows // trial_count
    # 重构为 (trial_count, time_points, num_neuron)
    reshaped = spike_counts.reshape(trial_count, time_points, num_neuron)
    # 对每个trial计算每个神经元的总放电数, 得到 shape (trial_count, num_neuron)
    trial_spike_sums = np.sum(reshaped, axis=1)
    
    non_zero_neurons = []
    for neuron in range(num_neuron):
        # 如果在所有trial中，该神经元的放电总数均大于0，则保留
        if np.all(trial_spike_sums[:, neuron] > 0):
            non_zero_neurons.append(neuron)
        else:
            print(f"Removing neuron {neuron} due to 0 spikes in at least one trial")
    # 筛选出保留的神经元列
    new_spike_counts = spike_counts[:, non_zero_neurons]
    
    # 返回新的神经元数量（trial_count和timeepoch保持不变）
    new_neuron_count = len(non_zero_neurons)
    
    return new_spike_counts, trial_count, timeepoch

# 将spike_cpunts模式转换为spike_times
def preprocess_spike_data(spike_counts, trial_count, timeepoch):
    """
    Preprocess spike count data into spiketimes format using vectorized operations.
    
    Parameters:
        spike_counts: numpy array of shape (trial_count * time_points, num_neuron).
                      Each entry is the number of spikes at that time point.
        trial_count:  int, number of trials.
        timeepoch:    list of tuples, each tuple (start, end) defines the time epoch for a trial.
    
    Returns:
        spiketimes: numpy array of shape (num_neuron, trial_count) with dtype=object.
                    Each entry is a 1D numpy array of spike times for that neuron on that trial.
        timeepoch:  the provided list of (start, end) tuples.
    """
    total_rows, num_neuron = spike_counts.shape
    time_points = total_rows // trial_count
    spiketimes = np.empty((num_neuron, trial_count), dtype=object)
    
    for trial_idx, (start, end) in enumerate(timeepoch):
        dt = (end - start) / time_points
        # Generate time points for this trial
        times = start + np.arange(time_points) * dt
        
        # Extract trial data of shape (time_points, num_neuron)
        trial_data = spike_counts[trial_idx * time_points:(trial_idx + 1) * time_points, :]
        
        # For each neuron, repeat the time points by the corresponding spike counts (vectorized over time)
        try:

            spiketimes[:, trial_idx] = np.array([
                np.repeat(times, trial_data[:, neuron].astype(int))
                for neuron in range(num_neuron)
            ], dtype=object)
        except ValueError as e:
            print(f"Error processing trial {trial_idx}: {e}")
    
    return spiketimes, timeepoch

# 与上面的不一样的
def preprocess_spike_data_1(spike_counts, trial_count, timeepoch):
    """
    Preprocess spike count data into spiketimes format using vectorized operations.
    If a given time bin contains multiple spikes, they are spread evenly within the dt interval.
    
    Parameters:
        spike_counts: numpy array of shape (trial_count * time_points, num_neuron).
                      Each entry is the number of spikes at that time point.
        trial_count:  int, number of trials.
        timeepoch:    list of tuples, each tuple (start, end) defines the time epoch for a trial.
    
    Returns:
        spiketimes: numpy array of shape (num_neuron, trial_count) with dtype=object.
                    Each entry is a 1D numpy array of spike times for that neuron on that trial.
        timeepoch:  the provided list of (start, end) tuples.
    """
    total_rows, num_neuron = spike_counts.shape
    time_points = total_rows // trial_count
    spiketimes = np.empty((num_neuron, trial_count), dtype=object)

    # Loop over trials
    for trial_idx, (start, end) in enumerate(timeepoch):
        dt = (end - start) / time_points
        # Create time bins: these represent the start time of each bin
        times = start + np.arange(time_points) * dt
        # Extract trial data of shape (time_points, num_neuron)
        trial_data = spike_counts[trial_idx * time_points:(trial_idx + 1) * time_points, :].astype(int)
        
        # Process each neuron; here we vectorize over time bins by using np.repeat plus a computed offset
        spiketimes_trial = []
        for neuron in range(num_neuron):
            counts = trial_data[:, neuron]
            if counts.sum() == 0:
                spiketimes_trial.append(np.array([]))
                continue
            # For every bin with nonzero spike count, generate evenly distributed offsets within dt.
            # This is done in a list comprehension over time bins.
            spike_times_neuron = np.concatenate([
                times[i] + dt * (np.arange(1, count + 1) / (count + 1))
                for i, count in enumerate(counts) if count > 0
            ])
            spiketimes_trial.append(spike_times_neuron)
        
        spiketimes[:, trial_idx] = np.array(spiketimes_trial, dtype=object)

    return spiketimes, timeepoch

def spiketimes_to_spikecounts(spiketimes, timeepoch, time_points=250):
    """
    Convert spiketimes format back to spike_counts.

    Parameters:
        spiketimes: numpy array of shape (num_neuron, trial_count), dtype=object.
                    Each entry is a 1D numpy array of spike times for that neuron on that trial.
        timeepoch:  list of tuples, each tuple (start, end) defines the time epoch for a trial.
        time_points: int, number of time bins per trial.

    Returns:
        spike_counts: numpy array of shape (trial_count * time_points, num_neuron).
    """
    if not isinstance(spiketimes, np.ndarray):
        spiketimes = np.array(spiketimes, dtype=object)

    num_neuron, trial_count = spiketimes.shape
    spike_counts = np.zeros((trial_count * time_points, num_neuron), dtype=int)

    for trial_idx, (start, end) in enumerate(timeepoch):
        dt = (end - start) / time_points
        bin_edges = np.linspace(start, end, time_points + 1)
        for neuron in range(num_neuron):
            spikes = spiketimes[neuron, trial_idx]
            if spikes.size == 0:
                continue
            # Digitize spike times into bins
            bin_idx = np.digitize(spikes, bin_edges) - 1
            # Remove spikes that fall outside the bins
            bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < time_points)]
            # Count spikes per bin
            counts, _ = np.histogram(bin_idx, bins=np.arange(time_points + 1) - 0.5)
            spike_counts[trial_idx * time_points:(trial_idx + 1) * time_points, neuron] = counts

    return spike_counts
