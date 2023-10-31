import numpy as np
import torch

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
        self.batch_index = [0] + self.batch_indices

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
        super().__init__(self, model, simulator, S, poses, vels, goal_poses, goal_vels, n_epochs, batch_size)
    
    def train(self):
        archit = self.model.archit
        crit = self.model.crit
        optim = self.model.optim

        # Implement this!!
        X_train = None
        Y_train = None
        S_train = self.S[0:self.n_train].copy()

        for epoch in range(self.n_epochs):
            epoch_indices = [int(i) for i in np.random.permutation(self.n_train)]

            for batch in range(self.n_batches):
                batch_indices = epoch_indices[self.batch_index[batch] : self.batch_index[batch+1]]
                X_batch = torch.tensor(X_train[batch_indices])
                Y_batch = torch.tensor(Y_train[batch_indices])
                S_batch = torch.tensor(S_train[batch_indices])
                # TODO: the remainder of this training loop (as well as validation)
