import math
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import torch

import gridWorld2DimDanger as env
import algorithm as algorithm
import pipeline as pipeline



def main():
    agent1 = algorithm.Agent(5, 'red',  None, None)
    agent2 = algorithm.Agent(5, 'green', None, None)
    agents = [agent1, agent2]
    world = env.TwoDimWorld(5,5, (4,3),agents)
    for i in range(len(world.agents)):
        world.agents[i].agentId = i
        world.agents[i].algorithm = algorithm.ActorCriticAlgorithm(len(agents), len(world.actionSpace), learningRate=0.01)

    

    numEpisode = 300
    totalRewards = pipeline.train(world, numEpisode)

    # plot rewards
    for agent in range(1, len(totalRewards)+1):
        plt.figure()
        agentKey = f'agent{agent}'
        rewards = totalRewards[agentKey]
        epochs = list(range(len(rewards)))
        plt.plot(epochs, rewards, label=f'Rewards for {agentKey}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Reward for {agentKey}')
        plt.show()


    rewards, allAgentTrajectories = pipeline.test(world)
    print(f"Total rewards: {rewards}")
    print(f"Trajectories: {allAgentTrajectories}")

    pipeline.plotPolicyFrame(allAgentTrajectories, world.goal, world.dangerousPositions)

    

if __name__ == "__main__":
    main()
    print("Done")

