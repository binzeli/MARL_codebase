
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class Actor(nn.Module):
    def __init__(self, numAgent, actionSize):
        super(Actor, self).__init__()
        self.numAgent = int(numAgent*2)
        self.actionSize = int(actionSize)
        self.linear1 = nn.Linear(self.numAgent, 16)
        self.linear2 = nn.Linear(16, self.actionSize)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = self.linear2(output)
        policy = F.softmax(output, dim=-1)
        return policy


class Critic(nn.Module):
    def __init__(self, numAgent):
        super(Critic, self).__init__()
        self.numAgent = int(numAgent*2)
        self.linear1 = nn.Linear(self.numAgent, 16)
        self.linear2 = nn.Linear(16, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        value = self.linear2(output)
        return value
    


class ActorCriticAlgorithm():
    def __init__(self, numAgentState, actionSize, learningRate):
        self.actor = Actor(numAgentState, actionSize)
        self.critic = Critic(numAgentState)
        self.optimizerActor = optim.Adam(self.actor.parameters(), lr=learningRate)
        self.optimizerCritic = optim.Adam(self.critic.parameters(), lr=learningRate)
        self.epsilon = 0.01
        self.minEpsilon = 0.01
        self.maxEpsilon = 1
        self.decayRate = 0.001
        self.gamma = 0.99
        self.policy = None
        self.actionProb = None


    def getCritic(self, state):
        return self.critic.forward(state)

        
    def getPolicy(self, states):
        output = self.actor(states)
        return output
    

    def updateEpsilon(self, episode):
        if episode % 100 == 0: 
            self.epsilon = self.minEpsilon + (self.maxEpsilon - self.minEpsilon) * np.exp(-self.decayRate * episode)


    
    def exploreOrExploit(self, episode):
        self.updateEpsilon(episode)
        if np.random.random() < self.epsilon:
            action_tensor = torch.distributions.Categorical(self.policy).sample()
        else:
            action_tensor = torch.argmax(self.policy, dim=-1, keepdim=True)  # Keep as tensor for indexing

        action_prob = torch.gather(self.policy, -1, action_tensor).squeeze(-1)  # Extract and remove extra dimensions
        return  action_tensor.item() , action_prob


    def chooseAction(self, state, agentId, episode):
        state = torch.tensor(state, requires_grad=True, dtype=torch.float).reshape(-1)
        self.policy = self.getPolicy(state)
        action, actionProb = self.exploreOrExploit(episode)
        self.actionProb = actionProb
        return action
    

    def getAdvantage(self, currentValue,nextValue,reward):
        return reward + self.gamma * nextValue - currentValue


    def updateActor(self, advantage):
        self.optimizerActor.zero_grad()
        actorLoss = -self.actionProb * advantage
        actorLoss.backward(retain_graph=True )
        self.optimizerActor.step()
        


    def updateCritic(self, reward, critic):
        self.optimizerCritic.zero_grad()
        criticLoss = F.mse_loss(reward, critic)
        criticLoss.backward()
        self.optimizerCritic.step()
        
        
    def updateActorCritic(self, currentValue,nextValue,reward):
        advantage = self.getAdvantage(currentValue,nextValue,reward)
        self.updateActor(advantage)
        self.updateCritic(reward, currentValue)
        

    
    def learn(self, allAgentCurrentStates, action, allAgentNextStates, reward):
        numAgent = len(allAgentCurrentStates)
        allAgentCurrentStates = torch.tensor(allAgentCurrentStates, requires_grad=True, dtype=torch.float).reshape(1, numAgent*2)
        allAgentNextStates = torch.tensor(allAgentNextStates, requires_grad=True, dtype=torch.float).reshape(1, numAgent*2)
        currentValue = self.getCritic(allAgentCurrentStates)
        nextValue = self.getCritic(allAgentNextStates)
        reward = torch.tensor(reward, dtype=torch.float, requires_grad=True)
        self.updateActorCritic(currentValue,nextValue,reward)


    def clearMemory(self):
        self.actionProb = None 



        
 



class Agent:
    def __init__(self, size, color, agentId, algorithm):
        self.size = size
        self.color = color  
        self.algorithm = algorithm
        self.agentId = agentId
        self.state = None
        self.reward = None
        self.actionSpace = None
        



    


 
             
