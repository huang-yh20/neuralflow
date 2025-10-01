import os
import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from analyse.py5_select_neurons import select_neurons_by_condition_difference

brain_region_list = ['left ALM', 'right ALM', 'left Medulla', 'right Medulla']

sys.path.append('../')
from preprocess.py12_spike_data_preprocess_utils import spiketimes_to_spikecounts

def visualize_temporal_firing_rates(data_path, brain_region, choice, task_name, output_dir, time_window=(-3.0, 2.0)):
    """
    Visualize temporal firing rate patterns for selected neurons under different conditions.

    Parameters: n 
        data_path: Path to the data file
        brain_region: Brain region to analyze
        choice: Choice parameter ('all_union', etc.)
        task_name: Task name for output naming
        output_dir: Directory to save output figures
        time_window: Time window for analysis (default: (-3.0, 2.0))
    """
    # Load data
    with open(data_path, 'rb') as f:
        data_ori = pkl.load(f)

    # Extract data for specific brain region
    choice_list_all = ['LL', 'LR', 'RL', 'RR']
    data = {}
    for choice_key in choice_list_all:
        data[choice_key] = data_ori[choice_key][brain_region]

    # Apply neuron selection based on condition difference (same as in py4_single_new_data_mini.py)
    time_slot = (-1.2, 0.0)  # Selection window
    num_neurons_ori = len(data['RR'])

    # Initial cleaning (same criteria as py4_single_new_data_mini.py)
    mini_batch_size = 20
    non_zero_neurons = []
    for neuron in range(num_neurons_ori):
        valid = True
        for choice_key in choice_list_all:
            spike_times = data[choice_key][neuron]
            non_empty_trials = sum(1 for trial in spike_times if len(trial) > 0)
            num_trials = len(spike_times)
            clean_thres = mini_batch_size if mini_batch_size < num_trials/2 else int(num_trials/2 - 1)
            if non_empty_trials < clean_thres:
                valid = False
                break
        if valid:
            non_zero_neurons.append(neuron)

    print(f"After initial cleaning: {len(non_zero_neurons)} neurons remain for {brain_region}")

    # Apply neuron selection based on condition difference
    temp_data_cleaned = {choice_key: [data[choice_key][neuron] for neuron in non_zero_neurons] for choice_key in choice_list_all}
    cleaned_to_original = {i: neuron_idx for i, neuron_idx in enumerate(non_zero_neurons)}
    selected_cleaned_indices = select_neurons_by_condition_difference(temp_data_cleaned, time_window=time_slot, num_neurons_to_select=20)
    selected_neurons = [cleaned_to_original[idx] for idx in selected_cleaned_indices]

    print(f"After neuron selection: {len(selected_neurons)} neurons selected for {brain_region}")

    if choice == 'all_union':
        # Create temporal visualizations for combined conditions
        conditions = {
            'Left (LL+LR)': ['LL', 'LR'],
            'Right (RL+RR)': ['RL', 'RR']
        }

        # Set up time parameters (matching the notebook)
        num_trials = len(data['LL'][0])
        time_epochs = [time_window] * num_trials
        time_points = 250  # Same as in the notebook
        dt = (time_window[1] - time_window[0]) / time_points
        time_axis = np.linspace(time_window[0], time_window[1], time_points)

        # Create figure with subplots for each selected neuron
        fig, axes = plt.subplots(5, 4, figsize=(16, 12))  # 5x4 grid for 20 neurons
        axes = axes.flatten()

        for neuron_idx, original_neuron_idx in enumerate(selected_neurons):
            ax = axes[neuron_idx]

            # Plot each combined condition
            for cond_name, cond_choices in conditions.items():
                # Combine spike data for this condition
                combined_data = []
                for choice_key in cond_choices:
                    combined_data.extend(data[choice_key][original_neuron_idx])

                # Convert to spike counts format
                combined_array = np.array(combined_data, dtype=object).reshape(1, -1)
                time_epochs_combined = [time_window] * len(combined_data)

                # Calculate spike counts and firing rates
                spike_counts = spiketimes_to_spikecounts(combined_array, time_epochs_combined, time_points)
                spike_counts_reshape = spike_counts.reshape((-1, len(combined_data)), order='F')
                firing_rates = spike_counts_reshape.mean(axis=1) * (1/dt)  # Convert to Hz
                firing_rates_std = spike_counts_reshape.std(axis=1) * (1/dt)

                # Plot with confidence intervals (similar to notebook)
                color = 'blue' if 'Left' in cond_name else 'red'
                ax.plot(time_axis, firing_rates, label=cond_name, color=color, linewidth=2)
                ax.fill_between(time_axis,
                               firing_rates - firing_rates_std,
                               firing_rates + firing_rates_std,
                               alpha=0.25, color=color)

            # Add reference lines (same as notebook)
            ax.axvline(x=0, color='grey', linestyle='--', alpha=0.7)
            ax.axvline(x=-1.85, color='grey', linestyle='--', alpha=0.7)
            ax.axvline(x=-1.2, color='grey', linestyle='--', alpha=0.7)

            ax.set_title(f'Neuron {original_neuron_idx + 1}', fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Firing Rate (Hz)', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

        plt.suptitle(f'Temporal Firing Rate Patterns - {brain_region}', fontsize=14)
        plt.tight_layout()

        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'temporal_firing_rates_{brain_region.replace(" ", "_")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved temporal visualization to {output_path}")

        # Also create a summary plot showing average temporal patterns across all selected neurons
        fig, ax = plt.subplots(figsize=(10, 6))

        for cond_name, cond_choices in conditions.items():
            all_neuron_rates = []

            for original_neuron_idx in selected_neurons:
                # Combine spike data for this condition
                combined_data = []
                for choice_key in cond_choices:
                    combined_data.extend(data[choice_key][original_neuron_idx])

                # Convert to spike counts format
                combined_array = np.array(combined_data, dtype=object).reshape(1, -1)
                time_epochs_combined = [time_window] * len(combined_data)

                # Calculate firing rates
                spike_counts = spiketimes_to_spikecounts(combined_array, time_epochs_combined, time_points)
                spike_counts_reshape = spike_counts.reshape((-1, len(combined_data)), order='F')
                firing_rates = spike_counts_reshape.mean(axis=1) * (1/dt)
                all_neuron_rates.append(firing_rates)

            # Calculate mean and std across neurons
            mean_rates = np.mean(all_neuron_rates, axis=0)
            std_rates = np.std(all_neuron_rates, axis=0)

            color = 'blue' if 'Left' in cond_name else 'red'
            ax.plot(time_axis, mean_rates, label=cond_name, color=color, linewidth=3)
            ax.fill_between(time_axis,
                           mean_rates - std_rates,
                           mean_rates + std_rates,
                           alpha=0.2, color=color)

        # Add reference lines
        ax.axvline(x=0, color='grey', linestyle='--', alpha=0.7)
        ax.axvline(x=-1.85, color='grey', linestyle='--', alpha=0.7)
        ax.axvline(x=-1.2, color='grey', linestyle='--', alpha=0.7)

        ax.set_title(f'Average Temporal Firing Rate Patterns - {brain_region}', fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Save summary figure
        summary_output_path = os.path.join(output_dir, f'avg_temporal_firing_rates_{brain_region.replace(" ", "_")}.png')
        plt.savefig(summary_output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved average temporal visualization to {summary_output_path}")

        return {
            'brain_region': brain_region,
            'selected_neurons': selected_neurons,
            'temporal_figure_path': output_path,
            'avg_temporal_figure_path': summary_output_path
        }

    else:
        print(f"Choice '{choice}' not implemented yet for temporal visualization")
        return None

def main():
    parser = argparse.ArgumentParser(description="Visualize temporal firing rate patterns for selected neurons.")
    parser.add_argument('--brain_region', required=True, choices=brain_region_list, help="Brain region parameter")
    parser.add_argument('--choice', default='all_union', choices=['all_union', 'all_seperate', 'LL', 'LR', 'RL', 'RR'], help="Choice parameter")
    parser.add_argument('--task_name', required=True, help="Task name for output naming")
    parser.add_argument('--output_dir', default='./figs/temporal_firing_rates', help="Output directory for figures")
    parser.add_argument('--time_window', nargs=2, type=float, default=[-3.0, 2.0], help="Time window for analysis (start end)")

    args = parser.parse_args()

    # Data path (same as in py4_single_new_data_mini.py)
    data_path = "./data/new_data_mini/session_057_aligned_spike_times.pkl"

    # Run visualization
    result = visualize_temporal_firing_rates(
        data_path=data_path,
        brain_region=args.brain_region,
        choice=args.choice,
        task_name=args.task_name,
        output_dir=args.output_dir,
        time_window=tuple(args.time_window)
    )

    if result:
        print(f"Temporal visualization completed successfully for {args.brain_region}")
        print(f"Selected neurons: {len(result['selected_neurons'])}")
        print(f"Individual neurons figure: {result['temporal_figure_path']}")
        print(f"Average patterns figure: {result['avg_temporal_figure_path']}")
    else:
        print("Temporal visualization failed")

if __name__ == "__main__":
    main()