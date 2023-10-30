from abc import ABC
import numpy as np

class PathPlanningController(ABC):
    def __call__(self, pos, goal_pos):
        pass

class CentralControllerStatic(PathPlanningController):
    # TODO: implement minimum velocity controller instead of minimum distances (should not be too different)
    def __init__(self, num_timesteps=64):
        self.num_timesteps = num_timesteps

    def __call__(self, pos, goal_pos):
        n_samples = pos.shape[0]; assert n_samples == goal_pos.shape[0]
        dim = pos.shape[1]; assert dim == goal_pos.shape[1]
        N = pos.shape[2]; assert N == goal_pos.shape[2]

        for samp in range(n_samples):
            pass
