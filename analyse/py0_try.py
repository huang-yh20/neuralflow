import os
import argparse
import scipy.io
import numpy as np
import pickle as pkl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from preprocess.py12_spike_data_preprocess_utils import preprocess_spike_data, clean_spike_data_1, clean_spike_data_2
import neuralflow
from neuralflow.feature_complexity.fc_base import FC_tools

# ---------- Setup: brain region list & argument parsing ----------
parser = argparse.ArgumentParser(description="Process brain_region and date parameters.")
parser.add_argument('--brain_region', required=True, help="Brain region parameter")
parser.add_argument('--date', required=True, help="Date parameter")
args = parser.parse_args()

brain_region = args.brain_region
date = args.date

# ---------- Load Data ----------
path = f"./data/raw_data/neural_data_{brain_region}/neural_data_{date}_{brain_region}.mat"
data = scipy.io.loadmat(path)
spike_counts_ori = data['region_data']
trial_counts = data['trial_count'].squeeze()
num_neurons_ori = spike_counts_ori.shape[1]

# ---------- Preprocessing ----------
# Reshape and select the proper time segment
spike_counts_reshape_ori = spike_counts_ori.reshape((-1, trial_counts, num_neurons_ori))
spike_counts_reshape = spike_counts_reshape_ori[57:150, :, :]
spike_counts = spike_counts_reshape.reshape((-1, num_neurons_ori))
time_epoch = [(0, 1.86)] * int(trial_counts)

# Use clean_spike_data_1 to remove extreme cases
spike_counts, trial_counts, time_epoch = clean_spike_data_1(
    spike_counts, trial_count=trial_counts, timeepoch=time_epoch
)
# Preprocess spike_counts to spiketimes format
spiketimes, timeepoch = preprocess_spike_data(spike_counts, trial_counts, time_epoch)

# ---------- Manual cleaning and experimental setup ----------
# Set the experimental trial count and number of neurons to use (like in py0_try.ipynb)
num_trials = 5
num_neurons = 5  # Use only the first 5 neurons for analysis
trial1 = np.arange(0, num_trials, 2)
trial2 = np.arange(1, num_trials, 2)

# Remove neurons that have no spikes in all trials of trial1 or trial2
non_zero_neurons = []
for neuron in range(num_neurons_ori):
    if not all(len(spiketimes[neuron, t]) == 0 for t in trial1) and \
       not all(len(spiketimes[neuron, t]) == 0 for t in trial2):
        non_zero_neurons.append(neuron)
spiketimes = spiketimes[non_zero_neurons, :]

# For the experiment, select only the first num_neurons neurons
spiketimes1 = spiketimes[0:num_neurons, trial1]
timeepoch1 = [timeepoch[i] for i in trial1]
data_1 = neuralflow.SpikeData(
    data=spiketimes1, dformat='spiketimes', time_epoch=timeepoch1, with_cuda=True
)
data_1.change_format('ISIs')

spiketimes2 = spiketimes[0:num_neurons, trial2]
timeepoch2 = [timeepoch[i] for i in trial2]
data_2 = neuralflow.SpikeData(
    data=spiketimes2, dformat='spiketimes', time_epoch=timeepoch2, with_cuda=True
)
data_2.change_format('ISIs')

# ---------- Optimization for data_1 ----------
grid = neuralflow.GLLgrid(Np=8, Ne=16, with_cuda=True)
init_model = neuralflow.model.new_model(
    peq_model={"model": "uniform", "params": {}},
    p0_model={"model": "cos_square", "params": {}},
    D=1,
    fr_model=[{"model": "linear", "params": {"slope": 1, "bias": 100}}] * num_neurons,
    params_size={'peq': 1, 'D': 1, 'fr': 1, 'p0': 1},
    grid=grid,
    with_cuda=True
)

optimizer = 'ADAM'
opt_params = {
    'max_epochs': 50,
    'mini_batch_number': 20,
    'params_to_opt': ['F', 'F0', 'D', 'Fr', 'C'],
    'learning_rate': {'alpha': 0.05}
}
ls_options = {
    'C_opt': {'epoch_schedule': [0, 1, 5, 30], 'nSearchPerEpoch': 3, 'max_fun_eval': 2},
    'D_opt': {'epoch_schedule': [0, 1, 5, 30], 'nSearchPerEpoch': 3, 'max_fun_eval': 25}
}
boundary_mode = 'absorbing'

optimization1 = neuralflow.optimization.Optimization(
    [data_1],
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
    fr_model=[{"model": "linear", "params": {"slope": 1, "bias": 100}}] * num_neurons,
    params_size={'peq': 1, 'D': 1, 'fr': 1, 'p0': 1},
    grid=grid,
    with_cuda=True
)

optimization2 = neuralflow.optimization.Optimization(
    [data_2],
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
with open(f"./saved/neural_data_{brain_region}_{date}_opt1.pkl", "wb") as f:
    pkl.dump(optimization1.results, f)

with open(f"./saved/neural_data_{brain_region}_{date}_opt2.pkl", "wb") as f:
    pkl.dump(optimization2.results, f)

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
output_folder = f"figs/{brain_region}_{date}"
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
    plt.savefig(os.path.join(output_folder, f'potential_condition_{cond}.png'))
    plt.close()

# Plot 4: p0 plot with different colors for data1 and data2
plt.figure()
opt_ind_1 = min_inds_1[FC_opt_ind]
opt_ind_2 = min_inds_2[FC_opt_ind]
p0_1 = optimization1.results['p0'][opt_ind_1][0]
p0_2 = optimization2.results['p0'][opt_ind_2][0]
plt.plot(init_model.grid.x_d, p0_1, linewidth=2, color=color_data1, label='Data1')
plt.plot(init_model.grid.x_d, (p0_2[::-1] if invert else p0_2), linewidth=2, color=color_data2, linestyle='--', label='Data2')
plt.xlabel('Latent state, x')
plt.ylabel('p₀(x)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'p0.png'))
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
    plt.ylabel(f'Firing rate f(x) [Hz] (Neuron {n+1})')
    plt.title(f'Neuron {n+1} Firing Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'neuron_{n+1}_firing_rate.png'))
    plt.close()
