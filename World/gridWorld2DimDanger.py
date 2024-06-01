import random

class TwoDimWorld:
    def __init__(self, rows, columns, goal, agents):
        self.rows = rows
        self.columns = columns
        self.goal = goal
        self.agents = agents
        # define state space
        self.stateSpace = [(row, column) for row in range(rows) for column in range(columns)]
        # define action space
        self.actionSpace = ['up', 'down', 'left', 'right', 'stay']
        self.updateAgentActionSpace() 

        # define dangerous position
        self.dangerousPositions = [(1, 1), (1, 3)]  # Example positions that are dangerous

        # Define the goal position
        self.goalPosition = (3, 3)

        # key dictionary for actions
        self.actionToIndex = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'stay': 4}
        # Inverting the dictionary to index to action
        self.indexToAction = {v: k for k, v in self.actionToIndex.items()}

############################################################################################################

    # Check if new state is within grid boundaries
    def isValidState(self, newState):
        if newState in self.stateSpace:
            return True
        else:
            return False
        


    
    # Random slip function when moving agent to a dangerous position
    def dangerousSlipMovement(self, state, newState):
        x, y = state
        if state in self.dangerousPositions:
            potential_slips = [
                    newState,                      # intended state
                    (x, min(y+1, self.columns-1)), # Slip up, cap at upper boundary
                    (x, max(y-1, 0)),              # Slip down, floor at lower boundary
                    (min(x+1, self.rows-1), y),    # Slip right, cap at right boundary
                    (max(x-1, 0), y)               # Slip left, floor at left boundary
            ]
            return random.choice(potential_slips)  # Randomly choose from potential slips
        else:
            return newState     # If not a dangerous position, return the intended new state
        
    # Transition function
    def transition(self, state, action):
        
        x, y = state
        action = self.indexToAction[action]
        if action == 'up':
            newState = (x, y+1)
        elif action == 'down':
            newState = (x, y-1)
        elif action == 'left':
            newState = (x-1, y)
        elif action == 'right':
            newState = (x+1, y)
        elif action == 'stay':
            newState = state
        else:
            return state # invalid action, remain in the same place

        if self.isValidState(newState):   # checks if the state is valid
            return self.dangerousSlipMovement(state, newState)
        return state  # Remain in the same place if move would go out of bounds
    
    
    # getSingleAgentNextState function
    def getSingleAgentNextState(self, state, action):
        singleAgentNextState = self.transition(state, action)
        return singleAgentNextState
    
    # getAllAgentRewards function
    def getAllAgentRewards(self, allAgentActions, allAgentNextStates):
        for i in range(len(self.agents)):
            if allAgentActions[i] == 'stay':
                self.agents[i].reward = 0
            elif allAgentNextStates.count(self.goal) == len(self.agents):
                self.agents[i].reward = 200
            elif allAgentNextStates[i] == self.goal:
                self.agents[i].reward = 50
            elif allAgentNextStates[i] in self.dangerousPositions:
                self.agents[i].reward = -10
            else:
                self.agents[i].reward = -1
        return [agent.reward for agent in self.agents]


    
############################################################################################################


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

    def getAllAgentActions(self, allAgentStates, episode):
        return [self.agents[i].algorithm.chooseAction(allAgentStates,self.agents[i].agentId, episode) for i in range(len(self.agents))]


    def getAllAgentNextStates(self, allAgentCurrentStates, allAgentActions):
        return [self.getSingleAgentNextState(state, action) for (state, action) in zip(allAgentCurrentStates, allAgentActions)]


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


