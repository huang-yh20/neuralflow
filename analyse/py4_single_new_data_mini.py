import os
import argparse
import scipy.io
import numpy as np
import pickle as pkl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from preprocess.py12_spike_data_preprocess_utils import preprocess_spike_data, clean_spike_data_1, clean_spike_data_2
from analyse.py5_select_neurons import select_neurons_by_condition_difference
import neuralflow
from neuralflow.feature_complexity.fc_base import FC_tools
brain_region_list = ['left ALM', 'right ALM', 'left Medulla', 'right Medulla']



# ---------- Setup: brain region list & argument parsing ----------

parser = argparse.ArgumentParser(description="Process brain_region and date parameters.")
parser.add_argument('--brain_region', required=True, choices = brain_region_list, help="Brain region parameter")
# parser.add_argument('--date', required=True, help="Date parameter")
parser.add_argument('--choice', default='all_union', choices=['all_union', 'all_seperate', 'LL', 'LR', 'RL', 'RR'], help="Choice parameter")
parser.add_argument('--alpha', default=0.05, help="Alpha parameter")
parser.add_argument('--task_name', required=True, help="task name")
parser.add_argument('--init_fr', default='unity_slope', help="initialization for firing rate")
parser.add_argument('--boundary', default='abs', help="boundary condition")
parser.add_argument('--select_neurons', action='store_true', help="Enable neuron selection based on condition difference")
parser.add_argument('--num_select_neurons', type=int, default=None, help="Number of neurons to select (default: use all)")
args = parser.parse_args()

brain_region = args.brain_region
task_name = args.task_name
alpha = float(args.alpha)
init_fr = args.init_fr
select_neurons = args.select_neurons
num_select_neurons = args.num_select_neurons

if args.boundary == 'abs':
    boundary_mode = 'absorbing'
elif args.boundary == 'ref':
    boundary_mode = 'reflecting'
else:
    raise ValueError

if init_fr == 'unity_slope':
    slope = 1
elif init_fr == 'constant':
    slope = 0
else:
    raise ValueError("init_fr must be either 'unity_slope' or 'constant'")

mini_batch_size = 20  # Define mini-batch size for neuron filtering
time_slot = (-1.2, 0.0) # 暂且固定

# ---------- Load Data ----------
path = "./data/new_data_mini/session_057_aligned_spike_times.pkl"
with open(path, 'rb') as f:
    data_ori = pkl.load(f)
# 取出单个脑区数据
choice_list_all = ['LL', 'LR', 'RL', 'RR']
data = {}
for choice in choice_list_all:
    data[choice] = data_ori[choice][brain_region]

num_neurons_ori = len(data['RR']) # 假设所有条件的神经元数量相同

# ---------- Preprocessing ----------
# 预处理，截取时间段
for choice in choice_list_all:
    for neuron in range(num_neurons_ori):
        trial_counts = len(data[choice][neuron])
        for trial in range(trial_counts):
            spike_times = data[choice][neuron][trial]
            spike_times = spike_times[(spike_times >= time_slot[0]) & (spike_times <= time_slot[1])]
            data[choice][neuron][trial] = spike_times

# 预处理，如果一个trial中所有神经元都不发放，则需要把这个trial清洗掉
for choice in choice_list_all:
    valid_trials = []
    trial_counts = len(data[choice][0])  # 假定所有神经元trial_counts都一样
    for trial in range(trial_counts):
        valid = False
        for neuron in range(num_neurons_ori):
            spike_times = data[choice][neuron][trial]
            if len(spike_times) > 0:
                valid = True
                break  # 只要有一个神经元有spike就可以
        if valid:
            valid_trials.append(trial)
    # 只保留valid_trials对应的trial
    for neuron in range(num_neurons_ori):
        data[choice][neuron] = [data[choice][neuron][trial] for trial in valid_trials]



# 清洗出可以进入运算的神经元，清洗条件比较严格，要求每个choice下都的非空trial数量不少于20个
choice_list_all = ['LL', 'LR', 'RL', 'RR']

non_zero_neurons = []
for neuron in range(num_neurons_ori):
    valid = True
    for choice in choice_list_all:
        spike_times = data[choice][neuron]
        non_empty_trials = sum(1 for trial in spike_times if len(trial) > 0)
        num_trials = len(spike_times)
        clean_thres = mini_batch_size if mini_batch_size < num_trials/2 else int(num_trials/2 - 1)
        if non_empty_trials < clean_thres:
            valid = False
            break
    if valid:
        non_zero_neurons.append(neuron)

print(f"After initial cleaning: {len(non_zero_neurons)} neurons remain")

# Apply neuron selection based on condition difference if enabled
if select_neurons:
    print("Applying neuron selection based on condition difference...")

    # First, create a temporary cleaned data structure for selection
    temp_data_cleaned = {choice: [data[choice][neuron] for neuron in non_zero_neurons] for choice in choice_list_all}

    # Create a mapping from cleaned neuron indices to original indices
    cleaned_to_original = {i: neuron_idx for i, neuron_idx in enumerate(non_zero_neurons)}

    # Select neurons from the cleaned set
    selected_cleaned_indices = select_neurons_by_condition_difference(
        temp_data_cleaned, time_window=time_slot, num_neurons_to_select=num_select_neurons
    )

    # Map back to original neuron indices
    non_zero_neurons = [cleaned_to_original[idx] for idx in selected_cleaned_indices]
    print(f"After neuron selection: {len(non_zero_neurons)} neurons selected")

data_cleaned = {choice: [data[choice][neuron] for neuron in non_zero_neurons] for choice in choice_list_all}

# Save selected neuron information (original indices from full dataset)
selected_neuron_info = {
    'original_neuron_indices': non_zero_neurons,
    'num_selected_neurons': len(non_zero_neurons),
    'selection_parameters': {
        'time_window': time_slot,
        'num_neurons_to_select': num_select_neurons,
        'brain_region': brain_region,
        'choice': args.choice,
        'task_name': task_name,
        'select_neurons': select_neurons
    }
}

# ---------- Combine data based on choice parameter ----------
# 如果是'all_union'，就把'LL', 'LR'合并，'RL', 'RR'合并
if args.choice == 'all_union':
    data_combined = [ [None]*len(non_zero_neurons), [None]*len(non_zero_neurons) ] # 两个条件，每个条件num_neurons个神经元
    for neuron in range(len(data_cleaned['LL'])):
        combined_trials_L = data_cleaned['LL'][neuron] + data_cleaned['LR'][neuron]
        combined_trials_R = data_cleaned['RL'][neuron] + data_cleaned['RR'][neuron]
        data_combined[0][neuron] = combined_trials_L
        data_combined[1][neuron] = combined_trials_R
elif args.choice == 'all_seperate':
    data_combined = [ [None]*len(non_zero_neurons) ] * 4  # 四个条件，每个条件num_neurons个神经元
    for neuron in range(len(non_zero_neurons)):
        for choice in choice_list_all:
            data_combined[choice_list_all.index(choice)][neuron] = data_cleaned[choice][neuron]
else:
    data_combined = [data_cleaned[args.choice]]

data_combined_array = []
for data_onechoice in data_combined:
    data_onechoice_array = np.array(data_onechoice, dtype=object)
    data_combined_array.append(data_onechoice_array)

# ---------- split trials and convert into spike data ----------
data_1_list, data_2_list = [], []
for data_idx, data_onechoice in enumerate(data_combined_array):
    num_trials_onechoice = len(data_onechoice[0])  # Assuming all neurons have the same number of trials
    trial1 = np.arange(0, num_trials_onechoice, 2)
    trial2 = np.arange(1, num_trials_onechoice, 2)

    spiketimes1 = data_onechoice[:, trial1]
    timeepoch1 = [time_slot] * len(trial1)
    data_1 = neuralflow.SpikeData(
        data=spiketimes1, dformat='spiketimes', time_epoch=timeepoch1, with_cuda=True
    )
    data_1.change_format('ISIs')    

    spiketimes2 = data_onechoice[:, trial2]
    timeepoch2 = [time_slot] * len(trial2)
    data_2 = neuralflow.SpikeData(
        data=spiketimes2, dformat='spiketimes', time_epoch=timeepoch2, with_cuda=True
    )
    data_2.change_format('ISIs') 

    data_1_list.append(data_1)
    data_2_list.append(data_2)   

num_neurons = len(non_zero_neurons) # 先暂时固定下来，以后再改
# ---------- Optimization for data_1 ----------
grid = neuralflow.GLLgrid(Np=8, Ne=16, with_cuda=True)
init_model = neuralflow.model.new_model(
    peq_model={"model": "uniform", "params": {}},
    p0_model={"model": "cos_square", "params": {}},
    D=1,
    fr_model=[{"model": "linear", "params": {"slope": slope, "bias": 100}}] * num_neurons,
    params_size={'peq': 1, 'D': 1, 'fr': 1, 'p0': len(data_1_list)},
    grid=grid,
    with_cuda=True
)

optimizer = 'ADAM'
opt_params = {
    'max_epochs': 50,
    'mini_batch_number': mini_batch_size,
    'params_to_opt': ['F', 'F0', 'D', 'Fr', 'C'],
    'learning_rate': {'alpha': alpha}
}
ls_options = {
    'C_opt': {'epoch_schedule': [], 'nSearchPerEpoch': 3, 'max_fun_eval': 2},
    'D_opt': {'epoch_schedule': [], 'nSearchPerEpoch': 3, 'max_fun_eval': 25}
} # 暂时不对结果进行line search优化

optimization1 = neuralflow.optimization.Optimization(
    data_1_list,
    init_model,
    optimizer,
    opt_params,
    ls_options,
    boundary_mode=boundary_mode,
    device='GPU'
)
print('Running optimization on datasample 1')
optimization1.run_optimization()

# ---------- Optimization for data_2 ----------
grid = neuralflow.GLLgrid(Np=8, Ne=16, with_cuda=True)
# Reuse num_neurons variable for consistency
init_model = neuralflow.model.new_model(
    peq_model={"model": "uniform", "params": {}},
    p0_model={"model": "cos_square", "params": {}},
    D=1,
    fr_model=[{"model": "linear", "params": {"slope": slope, "bias": 100}}] * num_neurons,
    params_size={'peq': 1, 'D': 1, 'fr': 1, 'p0': len(data_2_list)},
    grid=grid,
    with_cuda=True
)

optimization2 = neuralflow.optimization.Optimization(
    data_2_list,
    init_model,
    optimizer,
    opt_params,
    ls_options,
    boundary_mode=boundary_mode,
    device='GPU'
)
print('Running optimization on datasample 2')
optimization2.run_optimization()

# ---------- Save optimization results --------------
os.makedirs("./saved", exist_ok=True)
with open(f"./saved/neural_data_{args.choice}_{brain_region}_{57}_{task_name}_opt1.pkl", "wb") as f:
    pkl.dump(optimization1.results, f)

with open(f"./saved/neural_data_{args.choice}_{brain_region}_{57}_{task_name}_opt2.pkl", "wb") as f:
    pkl.dump(optimization2.results, f)

# Save selected neuron information (original indices from full dataset)
with open(f"./saved/neural_data_{args.choice}_{brain_region}_{57}_{task_name}_selected_neurons.pkl", "wb") as f:
    pkl.dump(selected_neuron_info, f)

# ---------- Feature Complexity and Consistency Analysis ----------
JS_thres = 0.0015
FC_stride = 5
smoothing_kernel = 10

fc = FC_tools(non_equilibrium=True, model=init_model, boundary_mode=boundary_mode, terminal_time=1)
FCs1, min_inds_1, FCs2, min_inds_2, JS, FC_opt_ind = fc.FeatureConsistencyAnalysis(
    optimization1.results, optimization2.results, JS_thres, FC_stride, smoothing_kernel
)
invert = fc.NeedToReflect(optimization1.results, optimization2.results)

# ---------- Plotting ----------
output_folder = f"figs/{args.choice}_{brain_region}_{57}_{task_name}"
os.makedirs(output_folder, exist_ok=True)

# Predefined colors for consistency
color_data1 = 'blue'
color_data2 = 'red'

# Plot 1: JS divergence vs. Feature Complexity
plt.figure()
plt.plot(FCs1, JS, linewidth=3, color=[120/255, 88/255, 170/255], label='JS curve')
plt.plot(FCs1[FC_opt_ind], JS[FC_opt_ind], '.', markersize=16, color='#87A2FB', label='Selected FC')
plt.plot(FCs1, JS_thres * np.ones_like(FCs1), '--', linewidth=1, color=[0.4]*3, label='JS threshold')
plt.xlabel('Feature complexity')
plt.ylabel('Jason-Shanon divergence')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'js_fc.png'))
plt.close()

# Plot 2: Negative relative loglikelihood (scaled) with different colors for data1 and data2
plt.figure()
for cond in range(1):
    ll1 = optimization1.results['logliks'][cond]
    ll10 = ll1[0]
    ll1 = (ll1 - ll10) / np.abs(ll10)
    ll2 = optimization2.results['logliks'][cond]
    ll20 = ll2[0]
    ll2 = (ll2 - ll20) / np.abs(ll20)
    iter_nums = np.arange(len(ll1), dtype='float64')
    plt.plot(iter_nums, ll1, color=color_data1, linewidth=2, label='Data1')
    plt.plot(iter_nums, ll2, color=color_data2, linestyle='--', linewidth=2, label='Data2')
plt.xlabel('Epoch number')
plt.ylabel('Relative loglikelihood')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'relative_loglikelihood.png'))
plt.close()

# Plot 3: Potential function (Phi) for each condition (here, only one) with distinct colors
for cond in range(1):
    plt.figure()
    opt_ind_1 = min_inds_1[FC_opt_ind]
    opt_ind_2 = min_inds_2[FC_opt_ind]
    Phi1 = -np.log(optimization1.results['peq'][opt_ind_1][cond]) * optimization1.results['D'][opt_ind_1][0]
    Phi2 = -np.log(optimization2.results['peq'][opt_ind_2][cond]) * optimization2.results['D'][opt_ind_2][0]
    plt.plot(init_model.grid.x_d, Phi1, linewidth=2, color=color_data1, label='Data1')
    plt.plot(init_model.grid.x_d, (Phi2[::-1] if invert else Phi2), linewidth=2, color=color_data2, linestyle='--', label='Data2')
    plt.xlabel('Latent state, x')
    plt.ylabel('Potential Φ(x)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'potential.png')) # 暂且只有一个势能
    plt.close()

# Plot 4: p0 plot with different colors for data1 and data2
num_conditions = len(data_1_list)
for cond in range(num_conditions):
    plt.figure()
    opt_ind_1 = min_inds_1[FC_opt_ind]
    opt_ind_2 = min_inds_2[FC_opt_ind]
    p0_1 = optimization1.results['p0'][opt_ind_1][cond]
    p0_2 = optimization2.results['p0'][opt_ind_2][cond]
    plt.plot(init_model.grid.x_d, p0_1, linewidth=2, color=color_data1, label='Data1')
    plt.plot(init_model.grid.x_d, (p0_2[::-1] if invert else p0_2), linewidth=2, color=color_data2, linestyle='--', label='Data2')
    plt.xlabel('Latent state, x')
    plt.ylabel('p₀(x)')
    plt.legend()
    plt.tight_layout()
    if num_conditions > 1:
        plt.savefig(os.path.join(output_folder, f'p0_condition_{cond}.png'))
    else:
        plt.savefig(os.path.join(output_folder, f'p0.png'))
    plt.close()

# Plot 5: Firing rate for each neuron with distinct colors for data1 and data2
fr_shape = optimization1.results['fr'][opt_ind_1].shape
num_neurons = fr_shape[2]
for n in range(num_neurons):
    plt.figure()
    fr1 = optimization1.results['fr'][opt_ind_1][0, :, n]
    fr2 = optimization2.results['fr'][opt_ind_2][0, :, n]
    plt.plot(init_model.grid.x_d, fr1, linewidth=2, color=color_data1, label='Data1')
    plt.plot(init_model.grid.x_d, (fr2[::-1] if invert else fr2), linewidth=2, color=color_data2, linestyle='--', label='Data2')
    plt.xlabel('Latent state, x')
    plt.ylabel(f'Firing rate f(x) [Hz] (Neuron {non_zero_neurons[n]})') # 从0开始编号
    plt.title(f'Neuron {non_zero_neurons[n]+1} Firing Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'neuron_{non_zero_neurons[n]}_firing_rate.png'))
    plt.close()
