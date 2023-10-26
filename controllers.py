from abc import ABC
import numpy as np
import itertools as it

class PathPlanningController(ABC):
    def __call__(self, poses, goal_poses):
        pass

class CentralControllerStatic(PathPlanningController):
    def __init__(self, num_timesteps=64):
        self.num_timesteps = num_timesteps

    def __call__(self, pos, goal_pos):
        n_samples = pos.shape[0]; assert n_samples == goal_pos.shape[0]
        dim = pos.shape[1]; assert dim == goal_pos.shape[1]
        N = pos.shape[2]; assert N == goal_pos.shape[2]

        assignments = np.eye(N).reshape(1, N, N)
        assignments = np.tile(assignments, (n_samples, 1, 1))
        perms = it.permutations(range(N))
        for i in range(n_samples):
            argmin = np.eye(N)
            for perm in perms:
                temp_assignment = np.eye(N)[perm,:]
                def distance(assignment):
                    return np.sum(np.linalg.norm(goal_pos[i,:,:] - pos[i,:,:] @ assignment, axis=1))
                if distance(temp_assignment) < distance(argmin):
                    argmin = temp_assignment
            assignments[i,:,:] = argmin

        vel = (goal_pos - pos @ assignments) / (self.num_timesteps - 1)
        poses = np.zeros(shape=(n_samples, self.num_timesteps, dim, N))
        vels = np.zeros(shape=(n_samples, self.num_timesteps, dim, N))
        poses[:,0,:,:] = pos
        vels[:,0,:,:] = vel
        for step in range(1, self.num_timesteps):
            vels[:,step,:,:] = vel
            poses[:,step,:,:] = poses[:,step-1,:,:] + vels[:,step,:,:]
        
        return poses, vels
