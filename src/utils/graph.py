import numpy as np
from scipy.spatial import distance_matrix

class AgentGraph:
    def __init__(self, agents, r_comm):
        self.agents = agents
        self.r_comm = r_comm
    
    def get_adjacency(self):
        agent_pos = np.array([agent.p for agent in self.agents])
        N = agent_pos.shape[0]
        S = distance_matrix(agent_pos, agent_pos)
        for i in range(N):
            for j in range(N):
                if i == j or S[i][j] > self.r_comm:
                    S[i][j] = 0
                else:
                    S[i][j] = 1

        return S
    
    def get_neighbors(self, agent):
        neighbors = np.nonzero(self.get_adjacency()[agent.i])
        return [a for a in self.agents if a.i in neighbors]
    
    def get_collision(self, r_agent):
        for agent in self.agents:
            for neighbor in self.get_neighbors(agent):
                if np.linalg.norm(neighbor.p - agent.p):
                    return True
        
        return False
