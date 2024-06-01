import math
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
from tqdm import tqdm
import os

def learnAllAgents(world, allAgentCurrentStates, allAgentActions, allAgentNextStates, allAgentRewards):
    for i in range(len(world.agents)): 
        world.agents[i].algorithm.learn(allAgentCurrentStates, allAgentActions[i], allAgentNextStates, allAgentRewards[i])

def train(world, numEpisode):
    totalRewards = {f'agent{i}': [] for i in range(1, len(world.agents) + 1)}
    
    for episode in tqdm(range(numEpisode), desc="Training Progress"):  # Wrap range with tqdm for a progress bar
        episodeRewards = [0] * len(world.agents)
        allAgentCurrentStates = world.getAllAgentInitialStates()
        while not world.isTerminal(allAgentCurrentStates):
            allAgentActions = world.getAllAgentActions(allAgentCurrentStates, episode)
            allAgentNextStates = world.getAllAgentNextStates(allAgentCurrentStates, allAgentActions)
            allAgentRewards = world.getAllAgentRewards(allAgentActions, allAgentNextStates)
            learnAllAgents(world, allAgentCurrentStates, allAgentActions, allAgentNextStates, allAgentRewards)
            allAgentCurrentStates = world.updateAllAgentStates(allAgentNextStates)

            episodeRewards = [episodeRewards[i] + allAgentRewards[i] for i in range(len(world.agents))]
        
        for i in range(1, len(world.agents) + 1):
            totalRewards[f'agent{i}'].append(episodeRewards[i-1])
    
    return totalRewards




def test(world):
    rewards = [0] * len(world.agents)
    allAgentCurrentStates = world.getAllAgentInitialStates()
    allAgentTrajectories = [allAgentCurrentStates]
    while not world.isTerminal(allAgentCurrentStates):
        print(allAgentCurrentStates)
        allAgentActions = world.chooseMaxRewardActions(allAgentCurrentStates)
        print(allAgentActions)
        allAgentNextStates = world.getAllAgentNextStates(allAgentCurrentStates, allAgentActions)
        allAgentRewards = world.getAllAgentRewards(allAgentActions, allAgentNextStates)
        allAgentCurrentStates = allAgentNextStates 

        rewards = [rewards[i] + allAgentRewards[i] for i in range(len(world.agents))]
        allAgentTrajectories.append(allAgentCurrentStates)
    return rewards, allAgentTrajectories



def plotAgentTrajectories(trajectory, goal):
    plt.figure(figsize=(8, 6))

    num_agents = len(trajectory[0])

    for agent in range(num_agents):
        agentX, agentY = zip(*[pos[agent] for pos in trajectory])
        #plt.plot(agentX, agentY, marker="o", label=f"Agent {agent + 1}")
        #plt.text(agentX[0], agentY[0], f"Start {agent + 1}", horizontalalignment='right')
        #plt.text(agentX[-1], agentY[-1], f"End {agent + 1}", horizontalalignment='left')
        offsetX = [x + 10 * (agent % 2 * 2 - 1) * (agent // 2) for x in agentX]
        offsetY = [y + 10 * (agent % 2 * 2 - 1) * (agent // 2) for y in agentY]

        plt.plot(offsetX, offsetY, marker="o", label=f"Agent {agent + 1}")
        plt.text(offsetX[0], offsetY[0], f"Start {agent + 1}", horizontalalignment='right')
        plt.text(offsetX[-1], offsetY[-1], f"End {agent + 1}", horizontalalignment='left')

    plt.scatter(*goal, color="red", zorder=5, s=100, label="Goal")
    plt.title("Trajectories of Agents")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.show()


def plotPolicyFrame(trajectory, goal, obstacles):
    os.makedirs('frames', exist_ok=True)
# Generate each frame
    for i in range(len(trajectory)):
        fig, ax = plt.subplots()
        ax.plot([p[0][0] for p in trajectory[:i+1]], [p[0][1] for p in trajectory[:i+1]], 'bo-', label='Agent 1')
        ax.plot([p[1][0] for p in trajectory[:i+1]], [p[1][1] for p in trajectory[:i+1]], 'ro-', label='Agent 2')

    # Mark the goal state
        ax.plot(goal[0], goal[1], 'go', markersize=10, label='Goal State')

    # Mark the dangerous states
        for idx, dangerous in enumerate(obstacles):
            ax.plot(dangerous[0], dangerous[1], 'rx', markersize=10, label=f'Dangerous State {idx + 1}')

    # Set up the grid, limits, and aspect of the axes
        ax.set_xticks(range(-1, 6))
        ax.set_yticks(range(-1, 6))
        ax.grid(True)
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_aspect('equal')

    # Add legend in the first frame
        if i == 0:
            ax.legend()

    # Save frame
        plt.savefig(f'frames/frame_{i:04d}.png')
        plt.close()





def plotCurrentPolicy (episode, numEpisode, world):
    if episode == 10 or episode == numEpisode-1:
            testRewards, allAgentTrajectories = test(world)
            plotAgentTrajectories(allAgentTrajectories, world.goal)





