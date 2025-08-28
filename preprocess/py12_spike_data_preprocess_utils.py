import numpy as np

def clean_spike_data(spike_counts, trial_count, timeepoch):
    # 遍历每一个trial, 如果某个trial的spike_counts全为0, 则将其剔除
    total_rows, num_neuron = spike_counts.shape
    time_points = total_rows // trial_count
    non_empty_trials = []
    for trial_idx in range(trial_count):
        if not np.all(spike_counts[trial_idx * time_points:(trial_idx + 1) * time_points, :] == 0):
            non_empty_trials.append(trial_idx)
        else:
            print(f"Removing empty trial: {trial_idx}")
    # 根据非空trial的索引，从原始数据中提取对应trial的数据（连续time_points行）
    filtered_trials = [
        spike_counts[trial_idx * time_points:(trial_idx + 1) * time_points, :]
        for trial_idx in non_empty_trials
    ]
    new_spike_counts = np.vstack(filtered_trials)
    
    # 如果timeepoch是列表，则只保留对应非空trial的时间段
    new_timeepoch = [timeepoch[i] for i in non_empty_trials]
    
    return new_spike_counts, new_timeepoch

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
