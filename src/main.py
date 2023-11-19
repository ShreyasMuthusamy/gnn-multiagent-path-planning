import numpy as np
import torch

from control.controllers import CentralControllerStatic
from control.model import AggregateGNN, Model
from control.trainers import TrainerPathPlanning
from simulation.sim import SwarmSimulator

# Parameters
n_samples = 240
N = 5
T = 64

controller = CentralControllerStatic(n_timesteps=T)
sim = SwarmSimulator(N, n_samples, n_timesteps=T, dynamic=False)
networks, poses, vels, goal_poses, goal_vels = sim.simulate(T, controller)

torch_state = torch.get_rng_state()
torch_seed = torch.initial_seed()
np_state = np.random.RandomState().get_state()

n_epochs = 50
batch_size = 20
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999

gnn_params = dict()
gnn_params['n_signals'] = [18, 16, 16]
gnn_params['n_filter_taps'] = [4]
gnn_params['bias'] = True
gnn_params['nonlinearity'] = torch.nn.ReLU
gnn_params['n_readout'] = [3]
gnn_params['n_edge_features'] = 1
gnn_params['n_timesteps'] = T
name = 'LocalGNN'

archit = AggregateGNN(**gnn_params)
optim = torch.optim.Adam(archit.parameters(), lr=learning_rate, betas=(beta1, beta2))
crit = torch.nn.MSELoss()
gnn = Model(archit, crit, optim, TrainerPathPlanning, name, 'experiments')
gnn.train(sim, networks, poses, vels, goal_poses, goal_vels, n_epochs, batch_size)

poses_test = poses[200:220][:,0,:,:]
goal_poses_test = goal_poses[200:220][:,0,:,:]
goal_vels_test = goal_vels[200:220][:,0,:,:]
poses_valid, _, goal_poses_valid, _ = sim.compute_trajectory(archit,
                                                             T,
                                                             poses_test,
                                                             goal_poses_test,
                                                             goal_vels_test)
cost = sim.cost(poses_valid, goal_poses_valid)
print(cost)
sim.animate(poses_valid, goal_poses_valid)
sim.animate(poses_test, goal_poses_test)