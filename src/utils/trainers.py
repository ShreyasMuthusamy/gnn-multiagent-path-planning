import numpy as np
import torch
import copy

from utils import state

class Trainer:
    def __init__(self,
               model,
               simulator,
               S,
               poses,
               vels,
               goal_poses,
               goal_vels,
               n_epochs,
               batch_size):
        self.model = model
        self.simulator = simulator
        self.S = S
        self.poses = poses
        self.vels = vels
        self.goal_poses = goal_poses
        self.goal_vels = goal_vels
        self.n_epochs = n_epochs

        self.n_train = 200
        self.n_test = 20
        self.n_valid = 20
        self.validation_interval = self.n_train // batch_size
        
        if self.n_train < batch_size:
            self.n_batches = 1
            self.batch_size = [self.n_train]
        elif self.n_train % batch_size != 0:
            self.n_batches = np.ceil(self.n_train / batch_size)
            self.batch_size = [batch_size] * (self.n_batches - 1)
            self.batch_size.append(self.n_train - sum(self.batch_size))
        else:
            self.n_batches = int(self.n_train / batch_size)
            self.batch_size = [batch_size] * self.n_batches
        
        self.batch_index = np.cumsum(self.batch_size).tolist()
        self.batch_index = [0] + self.batch_index

class TrainerPathPlanning(Trainer):
    def __init__(self,
               model,
               simulator,
               S,
               poses,
               vels,
               goal_poses,
               goal_vels,
               n_epochs,
               batch_size):
        super().__init__(model, simulator, S, poses, vels, goal_poses, goal_vels, n_epochs, batch_size)
    
    def train(self):
        archit = self.model.archit
        crit = self.model.crit
        optim = self.model.optim

        _, steps, _, N = self.poses.shape
        X_train = np.zeros((self.n_train, steps, 12, N))
        for step in range(steps):
            X_train[:,step,:,:] = state.signal(self.poses[0:self.n_train,step,:,:],
                                               self.goal_poses[0:self.n_train,step,:,:])
        Y_train = self.vels[0:self.n_train].copy()
        S_train = self.S[0:self.n_train].copy()

        for epoch in range(self.n_epochs):
            epoch_indices = [int(i) for i in np.random.permutation(self.n_train)]

            for batch in range(self.n_batches):
                ##############
                ## Training ##
                ##############

                batch_indices = epoch_indices[self.batch_index[batch] : self.batch_index[batch+1]]
                X_batch = torch.tensor(X_train[batch_indices]).float()
                Y_batch = torch.tensor(Y_train[batch_indices]).float()
                S_batch = torch.tensor(S_train[batch_indices]).float()
                
                archit.zero_grad()
                Y_pred = archit(X_batch, S_batch)
                loss = crit(Y_pred, Y_batch)
                loss.backward()
                optim.step()

                del X_batch
                del Y_batch
                del S_batch
                del loss

                ################
                ## Validation ##
                ################

                if (epoch * self.n_batches + batch) % self.validation_interval == 0:
                    pos_valid = self.poses[220:240,0,:,:]
                    goal_pos_valid = self.goal_poses[220:240,0,:,:]
                    goal_vel_valid = self.goal_vels[220:240,0,:,:]
                    poses_valid, _, goal_poses_valid, _ = self.simulator.compute_trajectory(archit,
                                                                                            steps,
                                                                                            pos_valid,
                                                                                            goal_pos_valid,
                                                                                            goal_vel_valid)
                    cost = self.simulator.cost(poses_valid, goal_poses_valid)[-1]
                    print(f'(E: {epoch + 1}, B: {batch + 1}), {cost}')

                    if epoch == 0 and batch == 0:
                        best_score = cost
                        best_epoch, best_batch = epoch, batch
                        self.model.save(label='Best')
                        # Start the counter
                    else:
                        valid_score = cost
                        if valid_score < best_score:
                            best_score = valid_score
                            best_epoch, best_epoch = epoch, batch
                            print(f'\t=> New best achieved: {best_score}')
                            self.model.save(label='Best')

        self.model.save(label='Last')
        # self.model.load(label='Best')
        if self.n_epochs > 0:
            print(f'\t=> Best validation achieved (E: {best_epoch + 1}, B: {best_batch + 1}): {best_score}')
