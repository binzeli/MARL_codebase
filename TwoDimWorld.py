## note: agent class should have agent.state, agent.reward, agent.actionSpace, agent.algorithm
import random


class TwoDimWorld:
    def __init__(self, rows, columns, goal, agents):
        self.rows = rows
        self.columns = columns
        self.goal = goal
        self.agents = agents
        self.stateSpace = [(row, column) for row in range(rows) for column in range(columns)]
        self.actionSpace = ['up', 'down', 'left', 'right', 'stay']
        self.updateAgentActionSpace() 
        # key dictionary for actions
        self.actionToIndex = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'stay': 4}
        # Inverting the dictionary to index to action
        self.indexToAction = {v: k for k, v in self.actionToIndex.items()}

    def updateAgentActionSpace(self):
        # store actionSpace in each agent
        for agent in self.agents:
            agent.actionSpace = self.actionSpace


    def getAllAgentInitialStates(self):
        initialStates = [(0, 0) for _ in range(len(self.agents))]
        for i in range(len(self.agents)):
            self.agents[i].state = initialStates
        return initialStates
        ## note: each agent.state contains all agents' current states

    '''def getAllAgentInitialStates(self):
        initialStates = []
        for agent in self.agents:
            while True:
                randomState = (random.randint(0, self.rows-1), random.randint(0, self.columns-1))
                if randomState != self.goal:
                    break
            agent.state = randomState
            initialStates.append(randomState)
        return initialStates'''

    def getAllAgentActions(self, allAgentStates, episode):
        return [self.agents[i].algorithm.chooseAction(allAgentStates,self.agents[i].agentId, episode) for i in range(len(self.agents))]


    def getAllAgentNextStates(self, allAgentCurrentStates, allAgentActions):
        return [self.getSingleAgentNextState(state, action) for (state, action) in zip(allAgentCurrentStates, allAgentActions)]



    def getAllAgentRewards(self, allAgentActions, allAgentNextStates):
        for i in range(len(self.agents)):
            if allAgentActions[i] == 'stay':
                self.agents[i].reward = 0
            elif allAgentNextStates.count(self.goal) == len(self.agents):
                self.agents[i].reward = 500
            elif allAgentNextStates[i] == self.goal:
                self.agents[i].reward = 20
            else:
                self.agents[i].reward = -1
        return [agent.reward for agent in self.agents]




    def getSingleAgentNextState(self, state, action):
        row, column = state
        action = self.indexToAction[action]
        if action == "up" and column < self.columns - 1:
            newSingleAgentState = (row, column + 1)
        elif action == "down" and column > 0:
            newSingleAgentState = (row, column - 1)
        elif action == "left" and row > 0:
            newSingleAgentState = (row - 1, column)
        elif action == "right" and row < self.rows - 1:
            newSingleAgentState = (row + 1, column)
        else:
            newSingleAgentState = state
        return newSingleAgentState
        

    def isTerminal(self, states):
        if any([state == self.goal for state in states]):
            for agent in self.agents:
                agent.algorithm.clearMemory()
            return True
        return False
    

    def updateAllAgentStates(self, allAgentNextStates):
        for agent, next_state in zip(self.agents, allAgentNextStates):
            agent.state = next_state
        return [agent.state for agent in self.agents]



    def chooseMaxRewardActions(self, allAgentCurrentStates):
        return [self.agents[i].algorithm.chooseBestAction(allAgentCurrentStates,self.agents[i].agentId) for i in range(len(self.agents))]

