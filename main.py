import gym
import numpy as np
import random
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")

# Number of episodes
number_of_episodes = 10000

# Max steps per episode
max_steps_per_episode = 100

# Learning rate
learning_rate = 0.1

# Discount factor
discount_factor = 0.99

# Exploration parameters
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# List of rewards
rewards_all_episodes = []

# Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# 2 For life or until learning is stopped
for episode in range(number_of_episodes):
    # Reset the environment and get first state
    state = env.reset()
    done = False
    rewards_current_episode = 0
    
    for step in range(max_steps_per_episode): 
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > exploration_rate:
            action = np.argmax(Q[state,:])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q-Table with new knowledge using Bellman Equation
        Q[state, action] = Q[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_factor * np.max(Q[new_state, :]))
        
        # Set state to be the new state
        state = new_state
        
        # Add new reward to episode reward
        rewards_current_episode += reward 
        
        if done == True: 
            break
            
    # Exploration rate decay   
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    
    # Add current episode reward to total rewards list
    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thosand_episodes = np.split(np.array(rewards_all_episodes),number_of_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("\n\n********Q-table********\n")
print(Q)
