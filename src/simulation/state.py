import numpy as np

def agent_network(poses, N, proximity=3):
    assert N == poses.shape[1]

    network = np.zeros((N, N))
    for i in range(N):
        pos_diff = poses - np.tile(poses[:,i].reshape(-1, 1), N)
        pos_diff = np.linalg.norm(pos_diff, axis=0)
        js = np.argsort(pos_diff)[1:proximity+1]
        network[i,js] = 1
        network[js,i] = 1
    
    assert np.array_equal(network, network.T)
    
    W = np.linalg.eigvalsh(network)
    max_eigenvalue = np.max(np.real(W))
    normalized_network = network / max_eigenvalue

    return network, normalized_network

def signal(poses, vels, goal_poses, goal_vels, agent_proximity=3, goal_proximity=3):
    n_samples = poses.shape[0]
    dim = poses.shape[1]
    N = poses.shape[2]

    signal = np.zeros((n_samples, 2 * agent_proximity + 2 * goal_proximity, N))
    for samp in range(n_samples):
        samp_signal = []

        for i in range(N):
            agent_signal = []

            pos_diff = poses[samp,:,:] - np.tile(poses[samp,:,i].reshape(-1, 1), N)
            pos_diff = np.linalg.norm(pos_diff, axis=0)
            indices = np.argsort(pos_diff)[1:agent_proximity+1]
            pos_diff = pos_diff[indices]
            agent_signal.append(pos_diff)

            vel_diff = vels[samp,:,indices] - np.tile(vels[samp,:,i].reshape(-1,1), agent_proximity)
            vel_diff = np.linalg.norm(vel_diff, axis=0)
            agent_signal.append(vel_diff)
        
            goal_diff = goal_poses[samp,:,:] - np.tile(poses[samp,:,i].reshape(-1, 1), N)
            goal_diff = np.linalg.norm(goal_diff, axis=0)
            indices = np.argsort(pos_diff)[:goal_proximity]
            goal_diff = goal_diff[indices]
            agent_signal.append(goal_diff)

            goal_vel_diff = goal_vels[samp,:,indices] - np.tile(goal_vels[samp,:,i].reshape(-1,1), goal_proximity)
            goal_vel_diff = np.linalg.norm(goal_vel_diff, axis=0)
            agent_signal.append(goal_vel_diff)

            agent_signal = np.hstack(agent_signal)
            assert agent_signal.shape == (2 * agent_proximity + 2 * goal_proximity,)
            samp_signal.append(agent_signal)
        
        samp_signal = np.vstack(samp_signal).T
        assert samp_signal.shape == (2 * agent_proximity + 2 * goal_proximity, N)
        signal[samp,:,:] = samp_signal
    
    return signal
