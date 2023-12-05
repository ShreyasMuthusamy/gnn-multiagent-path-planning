import gym
import matplotlib.pyplot as plt
import numpy as np

class Agent:
    def __init__(self, i):
        self.i = i
        self.s = None
        self.p = None
        self.v = None
        self.s_g = None

class ExpertController(gym.Env):
    def __init__(self, config):
        self.set_config(config)
    
    def set_config(self, config):
        self.times = config.sim_times
        self.state = None
        self.time_step = None

        self.total_time = self.times[-1]
        self.dt = self.times[1] - self.times[0]

        self.n_agents = config.n_agents
        self.state_dim_per_agent = 2 * config.dim
        self.action_dim_per_agent = config.dim
        self.r_agent = config.r_agent
        # self.r_obstacle = config.r_obstacle
        # self.r_obs_sense = config.r_obs_sense
        self.r_comm = config.r_comm

        self.a_min = config.a_min
        self.a_max = config.a_max
        self.v_min = config.v_min
        self.v_max = config.v_max

        self.n = self.state_dim_per_agent * self.n_agents
        self.m = self.action_dim_per_agent * self.n_agents

        self.agents = []
        for i in range(self.n_agents):
            self.agents.append(Agent(i))

        self.init_state_mean = 0.0
        self.init_state_var = 10.0

        if config.dim == 2:
            self.states_name = ['x-Position (m)',
                                'y-Position (m)',
                                'x-Velocity (m/s)',
                                'y-Velocity (m/s)']
            self.actions_name = ['x-Acceleration (m/s^2)',
                                 'y-Acceleration (m/s^2)']
        elif config.dim == 3:
            self.states_name = ['x-Position (m)',
                                'y-Position (m)',
                                'z-Position (m)',
                                'x-Velocity (m/s)',
                                'y-Velocity (m/s)',
                                'z-Velocity (m/s)']
            self.actions_name = ['x-Acceleration (m/s^2)',
                                 'y-Acceleration (m/s^2)',
                                 'z-Acceleration (m/s^2)']
        else:
            raise Exception('Illegal number of dimensions')
        
        self.config = config
        self.max_reward = 0

        # self.obstacles = []

    def render(self):
        # TODO: Implement this once the actual learning is done
        pass

    def step(self, a, compute_reward=True):
        self.s = self.next_state(self.s, a)
        d = self.done()

        if compute_reward:
            r = self.reward()
        else:
            r = 0

        self.time_step += 1

        return self.s, r, d, {}
    
    def done(self):
        for agent in self.agents:
            if np.linalg.norm(agent.s - agent.s_g) > 0.05:
                return False
        
        return True
    
    def observe(self):
        pass
