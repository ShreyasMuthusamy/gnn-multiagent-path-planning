import gym
import matplotlib.pyplot as plt
import numpy as np

from utils.graph import AgentGraph

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
        self.dim = config.dim
        self.state_dim_per_agent = 2 * self.dim
        self.action_dim_per_agent = self.dim
        self.r_agent = config.r_agent
        # self.r_obstacle = config.r_obstacle
        # self.r_obs_sense = config.r_obs_sense
        self.r_comm = config.r_comm
        self.graph = None
        self.gso = None

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

        if self.dim == 2:
            self.states_name = ['x-Position (m)',
                                'y-Position (m)',
                                'x-Velocity (m/s)',
                                'y-Velocity (m/s)']
            self.actions_name = ['x-Acceleration (m/s^2)',
                                 'y-Acceleration (m/s^2)']
        elif self.dim == 3:
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
        self.s = self.update(self.s, a)
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

    def reward(self):
        if self.graph.get_collision(self.r_agent):
            return -1
        
        return 0

    def reset(self):
        pass

    def update(self, s, a):
        s_next = np.zeros((self.n))
        dt = self.times[self.time_step+1]-self.times[self.time_step]

        for agent in self.agents:
            idx = self.state_dim_per_agent * agent.i
            p_idx = np.arange(idx, idx+self.dim)
            v_idx = np.arange(idx+self.dim, idx+self.dim)
            s_next[p_idx] = self.s[p_idx] + self.s[v_idx] * dt
            s_next[v_idx] = self.s[v_idx] + a[agent.i,:] * dt

            # Ensure physical limitations are met
            vel = np.linalg.norm(s_next[v_idx])
            if vel > self.v_max:
                s_next[v_idx] = s_next[v_idx] / vel * self.v_max
        
        self.update_agents(s_next)
        return s_next
    
    def update_agents(self, s):
        for agent in self.agents:
            idx = self.state_dim_per_agent * agent.i
            agent.p = s[idx:idx+self.dim]
            agent.v = s[idx+self.dim:idx+2*self.dim]
            agent.s = np.concatenate(agent.p, agent.v)

        self.graph = AgentGraph(self.agents, self.r_comm)
        self.gso = self.graph.get_adjacency()
