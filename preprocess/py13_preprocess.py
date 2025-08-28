import os
import scipy.io
from concurrent.futures import ThreadPoolExecutor, as_completed
from py12_spike_data_preprocess_utils import preprocess_spike_data, clean_spike_data

def process_file(brain_region, file_name):
    print(f"processing: {brain_region} - {file_name}")
    input_path = f"./data/raw_data/neural_data_{brain_region}/{file_name}"
    data = scipy.io.loadmat(input_path)
    spike_counts = data['region_data']
    trial_counts = int(data['trial_count'].squeeze())
    time_epoch = [(-3.0, 2.0)] * trial_counts
    # clean the spike data
    # 一定要记住自己做了什么操作，这里的操作在只拟合单个脑区的时候是可以的，但是如果同时拟合多个脑区，会出现trial对不上的情况
    spike_counts, time_epoch = clean_spike_data(spike_counts, trial_counts, time_epoch)
    # Preprocess the spike data
    spiketimes, timeepoch = preprocess_spike_data(spike_counts, trial_counts, time_epoch)
    # Save the processed data
    output_path = f"./data/processed_data/neural_data_{brain_region}/{file_name}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scipy.io.savemat(output_path, {'spiketimes': spiketimes, 'time_epoch': timeepoch})

def process_brain_region(brain_region):
    dir_path = f"./data/raw_data/neural_data_{brain_region}"
    entries = os.listdir(dir_path)
    tasks = []
    for file_name in entries:
        tasks.append((brain_region, file_name))
    return tasks

def main():
    brain_region_list = ['left_ALM', 'left_BLA', 'left_ECT', 'left_Medulla', 'left_Midbrain', 
                         'left_Striatum', 'left_Thalamus'] + \
                        ['right_ALM', 'right_BLA', 'right_ECT', 'right_Medulla', 
                         'right_Midbrain', 'right_Striatum', 'right_Thalamus']
    
    # Prepare a list of all file tasks
    all_tasks = []
    for region in brain_region_list:
        all_tasks.extend(process_brain_region(region))
    
    # Adjust max_workers based on your system
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_task = {executor.submit(process_file, region, fname): (region, fname) for region, fname in all_tasks}
        for future in as_completed(future_to_task):
            region, fname = future_to_task[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Error processing {region}-{fname}: {exc}")

if __name__ == '__main__':
    main()

