#!/usr/bin/env python3
"""
Temporary script to find and save selected neurons with specific parameters.
Creates selected_neurons_20_temp.pkl in saved/ directory.
"""

import os
import pickle as pkl
import numpy as np
import argparse

# Import the selection function from the existing script
import sys
sys.path.append('.')
from analyse.py5_select_neurons import select_neurons_by_condition_difference

def main():
    parser = argparse.ArgumentParser(description="Temp script to find selected neurons for all brain regions")
    parser.add_argument('--neuron_num', type=int, default=20, help="Number of neurons to select")
    parser.add_argument('--time_window', default='-1.2,0.0', help="Time window for analysis as 'start,end'")
    parser.add_argument('--data_path', default='./data/new_data_mini/session_057_aligned_spike_times.pkl', help="Path to data file")
    parser.add_argument('--output_dir', default='../saved', help="Output directory")

    args = parser.parse_args()

    # Parse time window
    time_window = tuple(map(float, args.time_window.split(',')))

    # List of all brain regions to process
    brain_regions = ['left ALM', 'right ALM', 'left Medulla', 'right Medulla']

    # Dictionary to store results for all brain regions
    all_regions_results = {}

    print(f"Loading data from: {args.data_path}")
    print(f"Time window: {time_window}")
    print(f"Target number of neurons: {args.neuron_num}")
    print(f"Processing brain regions: {brain_regions}")

    # Load data once
    with open(args.data_path, 'rb') as f:
        data_ori = pkl.load(f)

    choice_list_all = ['LL', 'LR', 'RL', 'RR']

    # Process each brain region
    for brain_region in brain_regions:
        print(f"\n{'='*50}")
        print(f"Processing brain region: {brain_region}")
        print(f"{'='*50}")

        # Extract data for current brain region
        data = {}
        for choice in choice_list_all:
            data[choice] = data_ori[choice][brain_region]

        num_neurons_ori = len(data['RR'])  # Assume all conditions have same number of neurons
        print(f"Original number of neurons in {brain_region}: {num_neurons_ori}")

        # Apply basic preprocessing (time window filtering)
        time_slot = time_window
        for choice in choice_list_all:
            for neuron in range(num_neurons_ori):
                trial_counts = len(data[choice][neuron])
                for trial in range(trial_counts):
                    spike_times = data[choice][neuron][trial]
                    spike_times = spike_times[(spike_times >= time_slot[0]) & (spike_times <= time_slot[1])]
                    data[choice][neuron][trial] = spike_times

        # Remove trials where no neurons fire
        for choice in choice_list_all:
            valid_trials = []
            trial_counts = len(data[choice][0])  # Assume all neurons have same trial count
            for trial in range(trial_counts):
                valid = False
                for neuron in range(num_neurons_ori):
                    spike_times = data[choice][neuron][trial]
                    if len(spike_times) > 0:
                        valid = True
                        break
                if valid:
                    valid_trials.append(trial)

            # Keep only valid trials
            for neuron in range(num_neurons_ori):
                data[choice][neuron] = [data[choice][neuron][trial] for trial in valid_trials]

        # Apply neuron selection
        print("Applying neuron selection...")
        selected_indices = select_neurons_by_condition_difference(
            data, time_window=time_window, num_neurons_to_select=args.neuron_num
        )

        print(f"Selected {len(selected_indices)} neurons")
        print(f"Selected neuron indices: {selected_indices}")

        # Create output data structure for this brain region
        selected_neuron_info = {
            'original_neuron_indices': selected_indices,
            'num_selected_neurons': len(selected_indices),
            'selection_parameters': {
                'brain_region': brain_region,
                'time_window': time_window,
                'num_neurons_to_select': args.neuron_num,
                'data_path': args.data_path
            },
            'full_data_info': {
                'original_total_neurons': num_neurons_ori,
                'brain_region': brain_region
            }
        }

        # Store results for this brain region
        all_regions_results[brain_region] = selected_neuron_info


    # Save combined results for all brain regions
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'selected_neurons_{args.neuron_num}_all_regions_temp.pkl')

    with open(output_file, 'wb') as f:
        pkl.dump(all_regions_results, f)

    print(f"\n{'='*60}")
    print(f"ALL BRAIN REGIONS PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Processed brain regions: {list(all_regions_results.keys())}")

    # Summary statistics across all regions
    for brain_region, results in all_regions_results.items():
        selected_count = results['num_selected_neurons']
        total_count = results['full_data_info']['original_total_neurons']
        print(f"{brain_region}: {selected_count}/{total_count} neurons selected")

if __name__ == "__main__":
    main()