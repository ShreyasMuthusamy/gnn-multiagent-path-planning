import numpy as np
from controllers import CentralControllerStatic
from sim import SwarmSimulator

if __name__ == '__main__':
    controller = CentralControllerStatic()
    sim = SwarmSimulator(5, 10, dim=3)
    sim.animate(controller)
