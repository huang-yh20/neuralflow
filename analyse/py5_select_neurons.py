import numpy as np

def select_neurons_by_condition_difference(data_cleaned, time_window=(-1.2, 0.0), num_neurons_to_select=None):
    """
    Select neurons based on their selectivity between LL and RR conditions.

    Parameters:
        data_cleaned: The cleaned data containing spike times for selected neurons
        time_window: Time window for analysis (default: (-1.2, 0.0))
        num_neurons_to_select: Number of neurons to select (default: use all)

    Returns:
        List of neuron indices sorted by selectivity (most selective first)
    """

    # Get data for LL and RR conditions from cleaned data
    data_LL = data_cleaned['LL']
    data_RR = data_cleaned['RR']

    num_neurons = len(data_LL)
    selectivity_ratios = []

    for neuron_idx in range(num_neurons):
        # Calculate firing rate for LL condition
        spike_times_LL = data_LL[neuron_idx]
        total_spikes_LL = 0
        for trial_spikes in spike_times_LL:
            spikes_in_window = trial_spikes[(trial_spikes >= time_window[0]) & (trial_spikes < time_window[1])]
            total_spikes_LL += len(spikes_in_window)

        # Calculate firing rate for RR condition
        spike_times_RR = data_RR[neuron_idx]
        total_spikes_RR = 0
        for trial_spikes in spike_times_RR:
            spikes_in_window = trial_spikes[(trial_spikes >= time_window[0]) & (trial_spikes < time_window[1])]
            total_spikes_RR += len(spikes_in_window)

        # Calculate average firing rates
        num_trials_LL = len(spike_times_LL)
        num_trials_RR = len(spike_times_RR)

        if num_trials_LL > 0 and num_trials_RR > 0:
            avg_rate_LL = total_spikes_LL / (num_trials_LL * (time_window[1] - time_window[0]))
            avg_rate_RR = total_spikes_RR / (num_trials_RR * (time_window[1] - time_window[0]))

            # Calculate selectivity ratio (max(LL/RR, RR/LL))
            if avg_rate_LL == 0 and avg_rate_RR == 0:
                ratio = 1.0  # No selectivity if both rates are zero
            elif avg_rate_RR == 0:
                ratio = float('inf')  # Infinite selectivity if RR rate is zero
            elif avg_rate_LL == 0:
                ratio = float('inf')  # Infinite selectivity if LL rate is zero
            else:
                ratio = max(avg_rate_LL / avg_rate_RR, avg_rate_RR / avg_rate_LL)
        else:
            ratio = 1.0  # No selectivity if no trials

        selectivity_ratios.append(ratio)

    # Sort neurons by selectivity ratio (descending order)
    sorted_indices = np.argsort(selectivity_ratios)[::-1]

    # Select top neurons if specified
    if num_neurons_to_select is not None and num_neurons_to_select < len(sorted_indices):
        selected_indices = sorted_indices[:num_neurons_to_select]
    else:
        selected_indices = sorted_indices

    return selected_indices