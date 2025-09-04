import os
from py12_spike_data_preprocess_utils import preprocess_spike_data, clean_spike_data
import neuralflow
import numpy as np
from neuralflow.feature_complexity.fc_base import FC_tools

# filepath: /work1/yuhan/neuralflow/preprocess/py0_try.py

import scipy.io
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# ---------- Setup: brain region list ----------
brain_region_list = ['left_ALM', 'left_BLA', 'left_ECT', 'left_Medulla', 'left_Midbrain', 'left_Striatum', 'left_Thalamus'] + \
                    ['right_ALM', 'right_BLA', 'right_ECT', 'right_Medulla', 'right_Midbrain', 'right_Striatum', 'right_Thalamus']

# ---------- Load Data and Preprocess ----------
path = f"./data/raw_data/neural_data_left_ALM/neural_data_0_left_ALM.mat"
data = scipy.io.loadmat(path)
spike_counts = data['region_data']
trial_counts = data['trial_count'].squeeze()
time_epoch = [(-3.0, 2.0)] * int(trial_counts)  # convert trial_count to integer time interval list

# Preprocess the spike data
spiketimes, timeepoch = preprocess_spike_data(spike_counts, trial_counts, time_epoch)

# ---------- Prepare first data sample ----------
spiketimes1, timeepoch1 = spiketimes[0:10, 0:50], timeepoch[0:50]
data_1 = neuralflow.SpikeData(
    data=spiketimes1, dformat='spiketimes', time_epoch=timeepoch1, with_cuda=True
)
data_1.change_format('ISIs')

# ---------- Prepare second data sample ----------
spiketimes2, timeepoch2 = spiketimes[0:10, 50:100], timeepoch[50:100]
data_2 = neuralflow.SpikeData(
    data=spiketimes2, dformat='spiketimes', time_epoch=timeepoch2, with_cuda=True
)
data_2.change_format('ISIs')

# ---------- Optimization for data_1 ----------
grid = neuralflow.GLLgrid(Np=8, Ne=16, with_cuda=True)
num_neurons = spiketimes2.shape[0]
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
num_neurons = spiketimes1.shape[0]
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
color_lines = ['#FF8C8C', [0.431, 0.796, 0.388], [1, 0.149, 0], [0, 0.561, 0]]

fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 4)

# Plot JS divergence vs. Feature Complexity
ax = plt.subplot(gs[0, 0])
ax.plot(FCs1, JS, linewidth=3, color=[120/255, 88/255, 170/255], label='JS curve')
ax.plot(FCs1[FC_opt_ind], JS[FC_opt_ind], '.', markersize=16, color='#87A2FB', label='Selected FC')
ax.plot(FCs1, JS_thres * np.ones_like(FCs1), '--', linewidth=1, color=[0.4]*3, label='JS threshold')
plt.legend()
plt.xlabel('Feature complexity')
plt.ylabel('Jason-Shanon divergence')

# Plot negative relative loglikelihood (scaled to start at 0)
ax = plt.subplot(gs[1, 0])
for cond in range(1):
    ll1 = optimization1.results['logliks'][cond]
    ll10 = ll1[0]
    ll1 = (ll1 - ll10) / np.abs(ll10)
    ll2 = optimization2.results['logliks'][cond]
    ll20 = ll2[0]
    ll2 = (ll2 - ll20) / np.abs(ll20)
    iter_nums = np.array(range(ll1.size)).astype('float64')
    ax.plot(iter_nums, ll1, color=color_lines[cond], linewidth=2, label=f'Condition {cond}')
    ax.plot(iter_nums, ll2, color=color_lines[cond], linewidth=2)
plt.legend()
plt.xlabel('Epoch number')
plt.ylabel('Relative loglikelihood')

opt_ind_1 = min_inds_1[FC_opt_ind]
opt_ind_2 = min_inds_2[FC_opt_ind]
for cond in range(1):
    ax = plt.subplot(gs[cond // 2, cond % 2 + 1])
    # Note: Original potential scaled back by D
    Phi1 = -np.log(optimization1.results['peq'][opt_ind_1][cond]) * optimization1.results['D'][opt_ind_1][0]
    Phi2 = -np.log(optimization2.results['peq'][opt_ind_2][cond]) * optimization2.results['D'][opt_ind_2][0]
    ax.plot(init_model.grid.x_d, Phi1, linewidth=2, color=color_lines[cond])
    # Reverse Phi2 if needed
    ax.plot(init_model.grid.x_d, Phi2[::-1] if invert else Phi2, linewidth=2, color=color_lines[cond])
    plt.xlabel('Latent state, $x$')
    plt.ylabel('Potential $\Phi(x)$')

ax = plt.subplot(gs[0, 3])
p0_1 = optimization1.results['p0'][opt_ind_1][0]
p0_2 = optimization2.results['p0'][opt_ind_2][0]
ax.plot(init_model.grid.x_d, p0_1, linewidth=2, color='black')
ax.plot(init_model.grid.x_d, p0_2[::-1] if invert else p0_2, linewidth=2, color='black')
plt.ylabel('$p_0(x)$')
plt.xlabel('Latent state, $x$')

# Plot firing rates for multiple neurons
fr_shape = optimization1.results['fr'][opt_ind_1].shape
num_fr_models = fr_shape[0]      # usually 1
num_grid_points = fr_shape[1]
num_neurons = fr_shape[2]

ax = plt.subplot(gs[1, 3])
colors = plt.cm.viridis(np.linspace(0, 1, num_neurons))
for n in range(num_neurons):
    fr1 = optimization1.results['fr'][opt_ind_1][0, :, n]
    fr2 = optimization2.results['fr'][opt_ind_2][0, :, n]
    ax.plot(init_model.grid.x_d, fr1, linewidth=2, color=colors[n],
            label=f'Neuron {n+1} data1' if n == 0 else None)
    ax.plot(init_model.grid.x_d, (fr2[::-1] if invert else fr2), linewidth=2,
            color=colors[n], linestyle='--',
            label=f'Neuron {n+1} data2' if n == 0 else None)
plt.xlabel('Latent state, $x$')
plt.ylabel('Firing rate $f(x)$, Hz')
plt.legend()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('neuralflow_results.png')
plt.close()