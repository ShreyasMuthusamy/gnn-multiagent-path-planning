import numpy as np

from control.controllers import CentralControllerStatic
from simulation.sim import SwarmSimulator

if __name__ == '__main__':
    controller = CentralControllerStatic(num_timesteps=3)
    sim = SwarmSimulator(5, 10, dim=3, num_timesteps=3)
    sim.animate(controller)
