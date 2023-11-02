from abc import ABC
import numpy as np
import scipy.optimize as opt

class PathPlanningController(ABC):
    def __call__(self, pos, goal_pos):
        pass

class CentralControllerStatic(PathPlanningController):
    def __init__(self, n_timesteps=64):
        self.n_timesteps = n_timesteps

    def __call__(self, pos, goal_pos):
        n_samples = pos.shape[0]; assert n_samples == goal_pos.shape[0]
        dim = pos.shape[1]; assert dim == goal_pos.shape[1]
        N = pos.shape[2]; assert N == goal_pos.shape[2]

        poses = np.zeros((n_samples, self.n_timesteps, dim, N))
        vels = np.zeros((n_samples, self.n_timesteps, dim, N))

        poses[:,0,:,:] = pos

        for samp in range(n_samples):
            samp_pos = pos[samp,:,:]
            samp_goal_pos = goal_pos[samp,:,:]
            cost = np.zeros((N, N))

            for agent in range(N):
                agent_pos = np.tile(samp_pos[:,agent].reshape(-1, 1), N)
                assert agent_pos.shape == (dim, N)
                pos_diff = samp_goal_pos - agent_pos
                pos_diff = np.power(np.linalg.norm(pos_diff, axis=0), 2)
                assert pos_diff.shape[0] == N
                cost[agent, :] = pos_diff
            
            row_ind, col_ind = opt.linear_sum_assignment(cost)
            for t in range(1, self.n_timesteps):
                vels[samp,t,:,:] = (samp_goal_pos[:,col_ind] - samp_pos[:,row_ind]) / (self.n_timesteps-1)
                poses[samp,t,:,:] += vels[samp,t,:,:] + poses[samp,t-1,:,:]
        
        return poses, vels
