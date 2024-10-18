import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import logging
import matplotlib.pyplot as plt
from city import CitySimulation

class DQN:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()
        self.rewards = []
        self.actions_taken = []
        self.q_values = []

    def create_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=self.state_dim))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_dim))
        model.compile(loss='mse', optimizer='adam')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.rewards.append(reward)
        self.actions_taken.append(action)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            self.q_values.append(np.max(target_f))
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Agent:
    def __init__(self, city, dqn):
        self.city = city
        self.dqn = dqn
        self.high_score_count = 0
    def act(self):
        state = self.get_state()
        action = self.dqn.act(state)
        self.adapt_city(action)
        return action

    def get_state(self):
        return np.array([
            self.city.count_empty_cells(),
            self.city.count_green_cells(),
            self.city.count_residential_cells(),
            self.city.count_industrial_cells(),
            self.city.count_commercial_cells(),
            self.city.population_growth_rate,
            self.city.environmental_impact_rate,
            self.city.infrastructure_development_rate,
            self.city.environmental_conservation_rate
        ]).reshape(1, -1)

    def adapt_city(self, action):
        if action == 0:  # increase_growth
            self.city.population_growth_rate += 0.08
        elif action == 1:  # reduce_impact
            self.city.environmental_impact_rate -= 0.1
        elif action == 2:  # improve_infrastructure
            self.city.infrastructure_development_rate += 0.08
        elif action == 3:  # decrease_pollution
            self.city.environmental_impact_rate -= 0.06
        elif action == 4:  # increase_housing_capacity
            self.city.population_growth_rate += 0.08
        elif action == 5:  # promote_commercial_growth
            self.city.infrastructure_development_rate += 0.1
        elif action == 6:  # stabilize_growth
            self.city.population_growth_rate += 0.09
            self.city.infrastructure_development_rate += 0.8
            self.city.environmental_impact_rate += 0.06

    def calculate_reward(self):
        city_value = self.city.sum_city_state_values()
        score = 100 - (abs(city_value - 50) / 50) * 100
        final_score = max(0, min(score, 100))
        if final_score > 90:
            self.high_score_count +=1
        return final_score


def plot_metrics(agent, dqn):
    # Plot rewards over time
    plt.figure(figsize=(12, 6))

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(dqn.rewards)
    plt.title('Rewards over Time')
    plt.xlabel('Generation')
    plt.ylabel('Reward')

    # Plot Q-values
    plt.subplot(1, 2, 2)
    plt.plot(dqn.q_values)
    plt.title('Q-values over Time')
    plt.xlabel('Generation')
    plt.ylabel('Q-value')

    plt.tight_layout()
    plt.show()
    plt.savefig('Q Values Over Time.png')  # Saves as a PNG file
    print("Plot saved as 'Q Values Over Time.png'")

    # Plot actions taken
    plt.figure(figsize=(12, 6))
    plt.hist(dqn.actions_taken, bins=dqn.action_dim, edgecolor='black')
    plt.title('Distribution of Actions Taken')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.show()
    plt.savefig('Distribution Of Actions Taken.png')  # Saves as a PNG file
    print("Plot saved as 'distribution_of_actions_taken.png'")


def plot_city(city):
    city_grid = city.grid  # Assuming city has a method to get the grid layout

    plt.imshow(city_grid, cmap='Blues')  # Visualize the grid as a heatmap
    plt.title("City State Visualization")
    plt.colorbar(label="Cell Value")
    plt.show()

def main():
    city = CitySimulation(20, 1)
    dqn = DQN(9, 7)  # State has 9 dimensions, 7 actions
    agent = Agent(city, dqn)

    for generation in range(1000):
        state = agent.get_state()
        action = agent.act()
        reward = agent.calculate_reward()
        next_state = agent.get_state()
        print(f'Agent Score :{reward}')
        done = False

        dqn.remember(state, action, reward, next_state, done)
        dqn.replay(32)
        dqn.decay_epsilon()

        city.evolve()

    plot_metrics(agent, dqn)

if __name__ == "__main__":
    main()
