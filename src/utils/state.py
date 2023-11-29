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

def signal(poses, goal_poses, goal_proximity=3):
    n_samples = poses.shape[0]
    dim = poses.shape[1]
    N = poses.shape[2]

    signal = np.zeros((n_samples, 3 * goal_proximity + 3, N))
    for samp in range(n_samples):
        samp_signal = []

        for i in range(N):
            agent_signal = []

            goal_diff = goal_poses[samp,:,:] - np.tile(poses[samp,:,i].reshape(-1, 1), N)
            goal_diff_norm = np.linalg.norm(goal_diff, axis=0)
            indices = np.argsort(goal_diff_norm)[:goal_proximity]
            goal_diff = np.hsplit(goal_diff[:,indices], goal_proximity)
            goal_diff = np.hstack([diff.T for diff in goal_diff])
            agent_signal.append(np.squeeze(goal_diff.T, 1))

            agent_signal.append(poses[samp,:,i].T)

            agent_signal = np.hstack(agent_signal).T
            assert agent_signal.shape == (3 * goal_proximity + 3,)
            samp_signal.append(agent_signal)
        
        samp_signal = np.vstack(samp_signal).T
        assert samp_signal.shape == (3 * goal_proximity + 3, N)
        signal[samp,:,:] = samp_signal
    
    return signal
