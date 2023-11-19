import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
import scipy.optimize as opt
from scipy.spatial import distance_matrix

from control.controllers import PathPlanningController
from simulation import state

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
                 n_timesteps=64,
                 dynamic=False):
        self.N = N
        self.n_samples = n_samples
        self.dim = dim
        self.max_agent_acceleration = max_agent_acceleration
        self.sensing_radius = sensing_radius
        self.n_timesteps = n_timesteps
        self.dynamic = dynamic
    
    def initial(self):
        init_pos = np.random.uniform(0, self.N, (self.n_samples, self.dim, self.N))
        goal_pos = np.random.uniform(self.N * 2, self.N * 3, (self.n_samples, self.dim, self.N))
        if self.dynamic:
            goal_vel = np.random.uniform(-self.N / self.n_timesteps, self.N / self.n_timesteps, (self.n_samples, self.dim, self.N))
        else:
            goal_vel = np.zeros((self.n_samples, self.dim, self.N))
        return init_pos, goal_pos, goal_vel
    
    def simulate(self, steps, controller: PathPlanningController):
        pos, goal_pos, goal_vel = self.initial()

        poses = np.zeros(shape=(self.n_samples, self.n_timesteps, self.dim, self.N))
        vels = np.zeros(shape=(self.n_samples, self.n_timesteps, self.dim, self.N))
        goal_poses = np.zeros(shape=(self.n_samples, self.n_timesteps, self.dim, self.N))
        goal_vels = np.zeros(shape=(self.n_samples, self.n_timesteps, self.dim, self.N))
        networks = np.zeros(shape=(self.n_samples, self.n_timesteps, self.N, self.N))

        poses, vels = controller(pos, goal_pos) if not self.dynamic else controller(pos, goal_pos, goal_vel)

        goal_poses[:,0,:,:] = goal_pos
        goal_vels[:,0,:,:] = goal_vel

        for samp in range(self.n_samples):
            _, networks[samp,0,:,:] = state.agent_network(pos[samp,:,:], self.N)

        for step in range(1, steps):
            goal_vels[:,step,:,:] = goal_vel
            goal_poses[:,step,:,:] = goal_poses[:,step-1,:,:] + goal_vels[:,step,:,:]
            for samp in range(self.n_samples):
                _, networks[samp,step,:,:] = state.agent_network(poses[samp,step,:,:], self.N)
            state.signal(poses[:,step,:,:], goal_poses[:,step,:,:])
        
        return networks, poses, vels, goal_poses, goal_vels
    
    def cost(self, pos, goal_pos):
        cost = 0
        for i in range(20):
            samp_pos = pos[i,-1,:,:]
            samp_goal_pos = goal_pos[i,-1,:,:]
            cost_matrix = distance_matrix(samp_pos.T, samp_goal_pos.T)
            row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
            pos_diff = samp_goal_pos[:,col_ind] - samp_pos[:,row_ind]
            cost += np.mean(np.linalg.norm(pos_diff, axis=1))
        
        return cost / self.n_samples

    def compute_trajectory(self, archit, steps, pos, goal_pos, goal_vel):
        batch_size = pos.shape[0]
        dim = pos.shape[1]
        N = pos.shape[2]
        poses = np.zeros((batch_size, steps, dim, N))
        vels = np.zeros((batch_size, steps, dim, N))
        goal_poses = np.zeros((batch_size, steps, dim, N))
        goal_vels = np.zeros((batch_size, steps, dim, N))
        signals = np.zeros((batch_size, steps, 18, N))
        graphs = np.zeros((batch_size, steps, N, N))

        poses[:,0,:,:] = pos
        goal_poses[:,0,:,:] = goal_pos
        goal_vels[:,0,:,:] = goal_vel

        for i in range(batch_size):
            _, graphs[i,0,:,:] = state.agent_network(pos[i], N)

        for t in range(1, steps):
            curr_pos = poses[:,t-1,:,:]
            curr_vel = vels[:,t-1,:,:]
            curr_goal_pos = goal_poses[:,t-1,:,:]
            curr_goal_vel = goal_vels[:,t-1,:,:]
            curr_signal = state.signal(curr_pos, curr_goal_pos)
            signals[:,t-1,:,:] = curr_signal

            X = torch.tensor(signals[:,0:t,:,:]).float()
            S = torch.tensor(graphs[:,0:t,:,:]).float()
            with torch.no_grad():
                Y = archit(X, S)
            vels[:,t,:,:] = Y.numpy()[:,-1,:,:]
            poses[:,t,:,:] = curr_pos + vels[:,t,:,:]
            goal_vels[:,t,:,:] = goal_vel
            goal_poses[:,t,:,:] = curr_goal_pos + goal_vels[:,t,:,:]
            for i in range(batch_size):
                _, graphs[i,t,:,:] = state.agent_network(poses[i,t,:,:], N)
        
        return poses, vels, goal_poses, goal_vels
    
    def animate(self, poses, goal_poses):
        last_agents = poses[-1,:,:,:]
        last_goals = goal_poses[-1,:,:,:]

        filenames = []

        for i in range(self.n_timesteps):
            plt.cla()

            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(last_agents[i,0,:], last_agents[i,1,:], last_agents[i,2,:], color='b')
            # ax.scatter(last_goals[i,0,:], last_goals[i,1,:], last_goals[i,2,:], color='r')
            # ax.set_xlim(-0.5 * self.N, 5.5 * self.N)
            # ax.set_ylim(-0.5 * self.N, 5.5 * self.N)
            # ax.set_zlim(-0.5 * self.N, 5.5 * self.N)

            plt.scatter(last_agents[i,0,:], last_agents[i,1,:], color='b')
            plt.scatter(last_goals[i,0,:], last_goals[i,1,:], color='r')
            plt.xlim(-0.5 * self.N, 3.5 * self.N)
            plt.ylim(-0.5 * self.N, 3.5 * self.N)

            plt.savefig(f'src/simulation/plots/snapshot{i+1}.png')
            filenames.append(f'src/simulation/plots/snapshot{i+1}.png')
        
        plt.close()
        
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('src/simulation/plots/sim.gif', images, fps=self.n_timesteps/2)
