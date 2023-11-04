import numpy as np

from control.controllers import CentralControllerDynamic
from simulation.sim import SwarmSimulator

if __name__ == '__main__':
    controller = CentralControllerDynamic()
    sim = SwarmSimulator(10, 1, dim=3, dynamic=True)
    sim.animate(controller)
