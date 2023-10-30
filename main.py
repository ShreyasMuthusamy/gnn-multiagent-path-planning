import numpy as np
from controllers import CentralControllerDynamic
from sim import SwarmSimulator

if __name__ == '__main__':
    controller = CentralControllerDynamic(num_timesteps=3)
    sim = SwarmSimulator(5, 10, dim=3, num_timesteps=3, dynamic=True)
    sim.animate(controller)
