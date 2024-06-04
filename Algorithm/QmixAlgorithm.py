#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import random
from collections import deque

class Agent:
    def __init__(self, agentId, network, targetNetwork, qmix):
        self.agentId = agentId
        self.network = network  # Agent-specific evaluation neural network
        self.targetNetwork = targetNetwork  # Agent-specific target neural network
        self.algorithm = qmix  # Reference to the QMix instance treated as 'algorithm'
        self.action = None
        self.state = None
        self.nextState = None
        self.reward = None

    def updateState(self, state, nextState, action, reward):
        self.state = state
        self.nextState = nextState
        self.action = action
        self.reward = reward

class AgentNetwork(nn.Module):
    def __init__(self, stateDim, hiddenDim, actionSpaceSize):
        super().__init__()
        self.actionSpaceSize = actionSpaceSize
        self.layers = nn.Sequential(
            nn.Linear(stateDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, actionSpaceSize)
        )

    def forward(self, x, action=None):
        qValues = self.layers(x)
        if action is not None:
            # print("qValues shape:", qValues.shape)
            # print("action shape:", action.shape)
            # print("action:", action)
            return qValues.gather(1, action.clone())  # Ensure action is cloned to avoid in-place modification
        return qValues

class MixingNetwork(nn.Module):
    def __init__(self, numAgents, stateDim):
        super().__init__()
        self.numAgents = numAgents
        self.stateDim = stateDim * numAgents  # Total number of state features from all agents
        self.hyperW1 = HyperNetwork(self.stateDim, numAgents * numAgents)  # HyperNetwork to generate a weight matrix for each agent interaction
        self.hyperW2 = HyperNetwork(self.stateDim, numAgents)  # HyperNetwork to generate a bias term for each agent

    def forward(self, qValues, state):
        stateFlat = state.view(-1, self.stateDim)
        # print("flattened state",stateFlat)
        # print("flattened state shape",stateFlat.shape)
        w1 = torch.abs(self.hyperW1(stateFlat)).clone().view(-1, self.numAgents, self.numAgents)
        b1 = torch.abs(self.hyperW2(stateFlat)).clone().view(-1, self.numAgents)

        # print("w1",w1)
        # print("w1 shape",w1.shape)
        # print("w2",b1)
        # print("w2 shape",b1.shape)
        if qValues.dim() == 3 and qValues.size(2) == 1:  # Handling joint values tensor
            # qValues = qValues.view(1, self.numAgents, 1)  # Reshape to (1, numAgents, 1)
            mixedQ = torch.bmm(w1, qValues).squeeze(2) + b1

        elif qValues.dim() == 3 and qValues.size(2) > 1:  # Handling joint next values tensor
            # qValues = qValues.view(1, self.numAgents, -1)  # Reshape to (1, numAgents, numActions)
            mixedQ = torch.bmm(w1, qValues).max(dim=2)[0] + b1

        else:
            raise ValueError("Unexpected qValues shape")

        return mixedQ

# w1.shape torch.Size([2, 64])
# b1.shape torch.Size([2, 32])
# w1.shape torch.Size([2, 2, 32])
# b1.shape torch.Size([2, 1, 32])
# qValues.unsqueeze(1).shape torch.Size([2, 1, 1])

class HyperNetwork(nn.Module):
    def __init__(self, stateDim, outputDim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(stateDim, 128),
            nn.ReLU(),
            nn.Linear(128, outputDim)
        )

    def forward(self, state):
        return self.network(state)
    
def orthogonalInit(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
            
class ExperienceReplayBuffer:
    def __init__(self, bufferSize):
        self.buffer = deque(maxlen=bufferSize)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batchSize):
        return random.sample(self.buffer, batchSize)

    def size(self):
        return len(self.buffer)
    
    def empty(self):
        self.buffer.clear()
    
    def printBuffer(self):
        for i, experience in enumerate(self.buffer):
            print(f"Experience {i}: {experience}")


class QMix:
    def __init__(self, networks, targetNetworks, mixingNetwork, targetMixingNetwork, epsilon, minEpsilon, maxEpsilon, decayRate, alpha, gamma, bufferSize=100000, batchSize=34):
        self.networks = networks
        self.targetNetworks = targetNetworks
        self.mixingNetwork = mixingNetwork
        self.targetMixingNetwork = targetMixingNetwork
        self.replayBuffer = ExperienceReplayBuffer(bufferSize)
        self.batchSize = batchSize
        self.numAgents = len(networks)
        self.epsilon = epsilon
        self.minEpsilon = minEpsilon
        self.maxEpsilon = maxEpsilon
        self.decayRate = decayRate
        self.alpha = alpha
        self.gamma = gamma
        self.trainStep = 0
        self.agentBuffer = []

        allParams = list(self.mixingNetwork.parameters())
        for network in self.networks:
            allParams += list(network.parameters())
        
        self.optimizer = optim.Adam(allParams, lr=self.alpha)
        
        self.lossHistory = {
            'agentNetworks': [],
            'mixingNetwork': []
        }
        
    def learn(self, state, action, nextState, reward, agentId):
        
        # agentCurrentState = state[agentId]
        # agentNextState = nextState[agentId]
        # agentCurrentStateTensor = torch.tensor([agentCurrentState], dtype=torch.float32) 
        # agentNextStateTensor = torch.tensor([agentNextState], dtype=torch.float32)
        agentCurrentState = state
        agentNextState = nextState
        agentCurrentStateTensor = torch.tensor(state, dtype=torch.float32).view(1, -1)
        agentNextStateTensor = torch.tensor(state, dtype=torch.float32).view(1, -1)
        actionTensor = torch.tensor([action], dtype=torch.long)
        outputValue = self.networks[agentId](agentCurrentStateTensor, actionTensor.unsqueeze(0)).detach()
        nextOutputValues = self.targetNetworks[agentId](agentNextStateTensor).detach()
        self.agentBuffer.append((outputValue, agentCurrentStateTensor, actionTensor.unsqueeze(0), agentNextStateTensor, nextOutputValues, reward, self.alpha, self.gamma))
        
        if len(self.agentBuffer) >= self.numAgents:
            self._processAgentBuffer()
        
    def _processAgentBuffer(self):
        outputs, currentStates, actions, nextStates, nextOutputs, rewards, alphas, gammas = zip(*self.agentBuffer)
        
        self.replayBuffer.add((outputs, currentStates, actions, nextStates, nextOutputs, rewards, alphas, gammas))
        self.agentBuffer = []

        if self.replayBuffer.size() >= self.batchSize:
            self._processReplayBuffer()
        
    def processStates(self, stateTuple):
        # Each element in stateTuple is a tuple of tensors for each agent
        # Convert the tuple of tensors to a list of tensors
        stateList = [torch.cat(s, dim=0) for s in stateTuple]

        # Stack the list of tensors along a new dimension to get shape (batchSize, numAgents, stateDim)
        processedState = torch.stack(stateList, dim=0)
        
        # print(processedState.shape)
        return processedState

    def processOutputs(self, outputTuple, nextOutput=False):
        # Convert the tuple of tensors to a list of tensors
        outputList = [torch.cat(o, dim=0) for o in outputTuple]

        if nextOutput:
            # Stack the list of tensors along a new dimension to get shape (batchSize, numAgents, numActions)
            processedOutput = torch.stack(outputList, dim=0)
        else:
            # Stack the list of tensors along a new dimension to get shape (batchSize, numAgents, 1)
            processedOutput = torch.stack(outputList, dim=0)
            
        # print(processedOutput.shape)
        return processedOutput

    def _processReplayBuffer(self):
        batch = self.replayBuffer.sample(self.batchSize)
        outputs, currentStates, actions, nextStates, nextOutputs, rewards, alphas, gammas = zip(*batch)

        outputs = self.processOutputs(outputs, nextOutput=False)
        nextOutputs = self.processOutputs(nextOutputs, nextOutput=True)
        currentStates = self.processStates(currentStates)
        nextStates = self.processStates(nextStates)
        # print("currentStates.shape",currentStates.shape)
        # print("nextStates",nextStates.shape)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        gammas = torch.tensor(gammas, dtype=torch.float32)

        maxNextQValues = torch.max(nextOutputs, dim=2)[0]
        # print("maxNextQValues.shape",maxNextQValues.shape)
        # print("maxNextQValues.unsqueeze(2)",maxNextQValues.unsqueeze(2).shape)
        maxTotalNextQValue = self.targetMixingNetwork(maxNextQValues.unsqueeze(2), nextStates)
        targetQValues = rewards + gammas * maxTotalNextQValue.detach()
        totalQValue = self.mixingNetwork(outputs, currentStates)

        tdError = totalQValue - targetQValues.unsqueeze(1)
        tdError2 = 0.5 * tdError.pow(2)

        mask = torch.ones_like(tdError2)
        maskedTdError = tdError2 * mask

        loss = maskedTdError.sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()  

        gradNorm = torch.nn.utils.clip_grad_norm_(self.mixingNetwork.parameters(),   max_norm=10)
        for network in self.networks:
            torch.nn.utils.clip_grad_norm_(network.parameters(),  max_norm=10)
        self.optimizer.step()

        self.lossHistory['agentNetworks'].append(tdError2.mean().item())
        self.lossHistory['mixingNetwork'].append(loss.item())

        if self.trainStep % 50 == 0:
            self.updateTargetNetworks()

        self.trainStep += 1

    def chooseAction(self, state, agentId, episode):
        self.updateEpsilon(episode)
        if random.random() < self.epsilon:
            return random.randint(0, self.networks[agentId].actionSpaceSize - 1)
        else:
            stateTensor = torch.tensor(state, dtype=torch.float32).view(1, -1)
            # stateTensor = torch.tensor(state[agentId], dtype=torch.float32).unsqueeze(0)
            qValues = self.networks[agentId](stateTensor)
            return torch.argmax(qValues, dim=1).item()
            
    def chooseMaxRewardAction(self, agent, state):    
        if isinstance(state[0], tuple):
            flattenedState = list(itertools.chain.from_iterable(state))
            stateTensor = torch.tensor([flattenedState], dtype=torch.float32)
        else:
            stateTensor = torch.tensor([state], dtype=torch.float32)
        qValues = agent.network(stateTensor)
        bestActionIdx = torch.argmax(qValues, dim=1).item()
        return bestActionIdx

    def getPolicy(self, state):
        actions = []
        for i, network in enumerate(self.networks):
            stateTensor = torch.tensor([state[i]], dtype=torch.float32)
            qValues = network(stateTensor)
            bestAction = torch.argmax(qValues, dim=1).item()
            actions.append(bestAction)
        return actions

    def updateEpsilon(self, episode):
        if episode % 100 == 0:
            self.epsilon = self.minEpsilon + (self.maxEpsilon - self.minEpsilon) * np.exp(-self.decayRate * episode)

    def lossHistory(self):
        return self.lossHistory
    
    def updateTargetNetworks(self):
        for targetNet, evalNet in zip(self.targetNetworks, self.networks):
            targetNet.load_state_dict(evalNet.state_dict())
        self.targetMixingNetwork.load_state_dict(self.mixingNetwork.state_dict())

