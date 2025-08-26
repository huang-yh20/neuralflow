'''
请帮我实现一个预处理spike数据的函数：
这个函数的输入是以下格式的spike data:
spiketimes format, where the data is numpy array of size
    (trial_count * time_points, num_neuron). In this case, all the trial have the same number of time points.
    The element of this array is the number of spikes at this time point.

这个函数的输出是以下格式的spike data:
spiketimes format, where the data is numpy array of size
        (num_neuron, N) of type object, N is the number of trials. Each of
        the entries is 1D array that specify spiketimes of each neuron on
        each trial. In this case time_epoch array specify each trial start
        and end times. For the example above, the spiketimes format would
        be the following:
            spiketimes = np.array(
                [
                    [np.array([], dtype=np.float64), np.array([0.05])],
                    [np.array([0.05]), np.array([3.05])],
                    [np.array([0.55]), np.array([1.05, 6.05])]
                    ],
                dtype=object
                )
            timeepoch = [(0, 1.55), (0, 10.05)]
'''
import numpy as np

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
        spiketimes[:, trial_idx] = np.array([
            np.repeat(times, trial_data[:, neuron].astype(int))
            for neuron in range(num_neuron)
        ], dtype=object)
    
    return spiketimes, timeepoch
