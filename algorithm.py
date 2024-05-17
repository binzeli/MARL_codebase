
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class Actor(nn.Module):
    def __init__(self, numAgentState, actionSize):
        super(Actor, self).__init__()
        self.numAgentState = int(numAgentState)
        self.actionSize = int(actionSize)
        self.linear1 = nn.Linear(self.numAgentState, 128)
        self.linear2 = nn.Linear(128, self.actionSize)

    def forward(self, state):
        #state = torch.from_numpy(state).float()  # Convert state to a Tensor
        #state = torch.tensor(state, dtype=torch.float)  # Convert state to a Tensor
        output = F.relu(self.linear1(state))
        output = self.linear2(output)
        policy = F.softmax(output, dim=-1)
        return policy


class Critic(nn.Module):
    def __init__(self, numAgentState):
        super(Critic, self).__init__()
        self.numAgentState = int(numAgentState)
        self.linear1 = nn.Linear(self.numAgentState, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, state):
        #state = torch.from_numpy(state).float()  # Convert state to a Tensor
        #state = torch.tensor(state, dtype=torch.float)  # Convert state to a Tensor
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
        self.gamma = 0.6
        self.policy = None
        self.actionProbList = []
        self.criticList = []
        self.rewardList = []


    def updateEpsilon(self, episode):
        if episode % 100 == 0: 
            self.epsilon = self.minEpsilon + (self.maxEpsilon - self.minEpsilon) * np.exp(-self.decayRate * episode)

        
    def getPolicy(self, state):
        ## input: a state in the form of a tensor
        ## output: a policy in tensor form
        return self.actor.forward((state))    # an tensor of length of action space
    

    def getExploredOrExploitedAction(self, episode):
        ## output: a single action
    
        self.updateEpsilon(episode)
        if np.random.random() < self.epsilon:
            action = torch.distributions.Categorical(self.policy).sample().item()
        else:
            action = torch.argmax(self.policy).item()
        return action, self.policy[action]

    

    def chooseAction(self, state, agentId, episode):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()  # Convert state to a Tensor
        self.policy = self.getPolicy(state)
        action, actionProb = self.getExploredOrExploitedAction(episode)
        self.actionProbList.append(actionProb)
        return action



    def getCritic(self, state):
        ## input: a state in the form of a tensor
        ## output: a value in tensor form
        return self.critic.forward((state))     


    def normalizeReward(self):
        self.rewardList = torch.tensor(self.rewardList, dtype=torch.float)  # Convert rewardList to a Tensor
        returns = torch.zeros_like(self.rewardList)
        discountedSum = 0.0
        for i in reversed(range(len(self.rewardList))):
                discountedSum = self.rewardList[i] + self.gamma * discountedSum
                returns[i] = discountedSum
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        self.rewardList = returns
    


    def getAdvantage(self):
        self.normalizeReward()
        return self.rewardList - self.criticList


    def updateActor(self, advantage):
        self.optimizerActor.zero_grad()
        self.actionProbList = torch.stack(tuple(self.actionProbList), 0)  # Convert actionProbList to a Tensor
        actorLoss = -(torch.sum(torch.log(self.actionProbList) * advantage))
        actorLoss.backward(retain_graph=True)
        self.optimizerActor.step()


    def updateCritic(self):
        self.optimizerCritic.zero_grad()
        self.criticList = torch.stack(tuple(self.criticList), 0) 
        self.rewardList = torch.stack(tuple(self.rewardList), 0)  
        lossFunction = torch.nn.SmoothL1Loss()  
        criticLoss = lossFunction(self.rewardList.view(-1), self.criticList.view(-1)) 
        criticLoss.backward() 
        self.optimizerCritic.step()
        
        


    def updateActorCritic(self):
        advantage = self.getAdvantage(self.criticList, self.rewardList)
        self.updateActor(advantage)
        self.updateCritic()

    


        
  
    def learn(self, allAgentCurrentStates, action, allAgentNextStates, reward):
        if not isinstance(allAgentCurrentStates, torch.Tensor):
            allAgentCurrentStates = torch.from_numpy(allAgentCurrentStates).float()  # Convert state to a Tensor
        self.criticList.append(self.getCritic(allAgentCurrentStates))
        self.rewardList.append(reward)
        self.updateActorCritic()


        
 



class Agent:
    def __init__(self, size, color, algorithm):
        self.size = size
        self.color = color  
        self.algorithm = algorithm
        


        

    def getSingleMaxRewardAction(self):
        if not isinstance(self.state, torch.Tensor):
            self.state = torch.from_numpy(self.state).float()  # Convert state to a Tensor
        self.policy = self.algorithm.getPolicy(self.state)
        self.action = torch.argmax(self.policy).item()


    


 
             