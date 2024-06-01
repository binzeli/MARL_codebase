import math
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import gym
import torch


class World:
    def __init__(self, agents):
        self.env = gym.make('CartPole-v1')

        self.agents = agents
        self.stateSize = self.env.observation_space.shape[0]
        self.actionSize = self.env.action_space.n


    # initial state for all agents
    def getAllAgentInitialStates(self):
        for i in range(len(self.agents)):
            self.agents[i].state = self.env.reset()[0]
        return [agent.state for agent in self.agents]
 
    # actions for all agents
    def chooseActions(self, allAgentCurrentStates, episode):
        for i in range(len(self.agents)):
            self.agents[i].getSingleAgentAction(episode)
        return [agent.action for agent in self.agents]
    

    # transition for all agents
    def transition(self, allAgentCurrentStates, allAgentActions):
        for i in range(len(self.agents)):
            self.singleAgentTransition(self.agents[i], allAgentActions[i])
        return [agent.nextState for agent in self.agents]


    # rewards for all agents
    def getAllAgentRewards(self, allAgentActions, allAgentNextStates):
        for agent in self.agents:
            self.getSingleAgentReward(agent)
        return [agent.reward for agent in self.agents]


    # reward for single agent
    def getSingleAgentReward(self, agent):
        return agent.reward
        

    # transition for single Agent
    def singleAgentTransition(self, agent, action):
        nextState, reward, done, _ =  self.env.step(action)[0:4]
        agent.nextState = nextState
        agent.reward = reward
        agent.terminal = done
        

    
    def isTerminal(self, allAgentCurrentStates):
        if any(agent.terminal == True for agent in self.agents):
            for agent in self.agents: 
                agent.normalizeReward()
                agent.criticList = torch.stack(tuple(agent.criticList), 0)  # Convert criticList to a Tensor
                agent.algorithm.updateActorCritic(agent.criticList, agent.rewardList, agent.actionProbList) 
                agent.terminal = False
                agent.rewardList = []
                agent.actionProbList = []
                agent.criticList = []

            return True
        else:
            return False
        

    def isTerminalTest(self):
        if any(agent.terminal == True for agent in self.agents):
            return True
        else:
            return False
    

    def chooseMaxRewardActions(self, allAgentCurrentStates):
        for i in range(len(self.agents)):
            self.agents[i].getSingleMaxRewardAction()
        return [agent.action for agent in self.agents]
        

