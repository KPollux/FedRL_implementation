# %%
import gym
import numpy
import random
from os import system, name
from time import sleep

"""Setup"""

# %%
env = gym.make("Taxi-v3").env # Setup the Gym Environment

# Make a new matrix filled with zeros.
# The matrix will be 500x6 as there are 500 states and 6 actions.
q_table = numpy.zeros([env.observation_space.n, env.action_space.n])

training_episodes = 5000 # Amount of times to run environment while training.
display_episodes = 10 # Amount of times to run environment after training.

# Hyperparameters
alpha = 0.1 # Learning Rate
gamma = 0.6 # Discount Rate
epsilon = 0.1 # Chance of selecting a random action instead of maximising reward.

# For plotting metrics
all_epochs = []
all_penalties = []

sum_list = []

"""Training the Agent"""

for i in range(training_episodes):
    state = env.reset()[0] # Reset returns observation state and other info. We only need the state.
    done = False
    log_rewards = 0
    penalties, reward, = 0, 0
    
    t = 0
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Pick a new action for this state.
        else:
            action = numpy.argmax(q_table[state]) # Pick the action which has previously given the highest reward.

        next_state, reward, truncated, terminated, info = env.step(action) 
        done = truncated or terminated 
        
        old_value = q_table[state, action] # Retrieve old value from the q-table.
        next_max = numpy.max(q_table[next_state])

        # Update q-value for current state.
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10: # Checks if agent attempted to do an illegal action.
            penalties += 1

        state = next_state

        log_rewards += reward

        t += 1
        if t > 100: # Prevents agent from getting stuck in an infinite loop.
            break
    
    sum_list.append(log_rewards)
    if i % 100 == 0: # Output number of completed episodes every 100 episodes.
        print(f"Episode: {i}")

print("Training finished.\n")


# %%
import matplotlib.pyplot as plt
import seaborn as sns

def moving_average(data_list, window_size=100):
    moving_averages = []
    cumsum = [0]
    for i, x in enumerate(data_list, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=window_size:
            moving_aver = (cumsum[i] - cumsum[i-window_size])/window_size
            moving_averages.append(moving_aver)
    return moving_averages

sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.figure(figsize=(15, 7.5))
sum_list_ = moving_average(sum_list)
plt.plot(sum_list_)
plt.xlabel('episode num')
plt.ylabel('points')
plt.show()
# %%
len(sum_list)
# %%
