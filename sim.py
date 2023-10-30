import numpy as np
import matplotlib.pyplot as plt

from controllers import PathPlanningController

class SwarmSimulator:
    '''
    Simulator for the swarm of robots
    - Used for path planning, but can be used for similar applications (e.g. coverage control)
    '''
    def __init__(self,
                 N,
                 n_samples,
                 dim=3,
                 max_agent_acceleration=1.0,
                 sensing_radius=1.0,
                 num_timesteps=64,
                 dynamic=False):
        self.N = N
        self.n_samples = n_samples
        self.dim = dim
        self.max_agent_acceleration = max_agent_acceleration
        self.sensing_radius = sensing_radius
        self.num_timesteps = num_timesteps
        self.dynamic = dynamic
    
    def initial(self):
        init_pos = np.random.uniform(0, self.N * 5, (self.n_samples, self.dim, self.N))
        init_vel = np.zeros((self.n_samples, self.dim, self.N))
        goal_pos = np.random.uniform(0, self.N * 5, (self.n_samples, self.dim, self.N))
        goal_vel = np.zeros((self.n_samples, self.dim, self.N)) # if !self.dynamic
        return init_pos, init_vel, goal_pos, goal_vel
    
    def simulate(self, steps, controller: PathPlanningController):
        pos, vel, goal_pos, goal_vel = self.initial()

        poses = np.zeros(shape=(self.n_samples, self.num_timesteps, self.dim, self.N))
        vels = np.zeros(shape=(self.n_samples, self.num_timesteps, self.dim, self.N))
        goal_poses = np.zeros(shape=(self.n_samples, self.num_timesteps, self.dim, self.N))
        goal_vels = np.zeros(shape=(self.n_samples, self.num_timesteps, self.dim, self.N))

        poses, vels = controller(pos, goal_pos) # if not self.dynamic else controller(pos, goal_pos, goal_vel)

        goal_poses[:,0,:,:] = goal_pos
        goal_vels[:,0,:,:] = goal_vel

        for step in range(1, steps):
            goal_vels[:,step,:,:] = goal_vel
            goal_poses[:,step,:,:] = goal_poses[:,step-1,:,:] + goal_vels[:,step,:,:]
        
        return poses, vels, goal_poses, goal_vels
    
    def animate(self, controller: PathPlanningController):
        poses, vels, goal_poses, goal_vels = self.simulate(self.num_timesteps, controller)
        last_agent = poses[-1,:,:,:]
        last_goal = goal_poses[-1,:,:,:]

        for i in range(self.num_timesteps):
            plt.scatter(last_agent[i,0,:], last_agent[i,1,:], color='b')
            plt.scatter(last_goal[i,0,:], last_goal[i,1,:], color='r')
            plt.show()
