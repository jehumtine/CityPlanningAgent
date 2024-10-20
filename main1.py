import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from keras.src.layers import Dense
from keras.src.models import Sequential
import random
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from city import CitySimulation
import pygame
import sys

CELL_SIZE = 20
GRID_SIZE = 20
FPS = 10

pygame.init()

# Set up display
width = GRID_SIZE * CELL_SIZE + 400
height = GRID_SIZE * CELL_SIZE + 150
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("City Evolution Simulation")
colors = {
    "E": (255, 255, 255),  # White for Empty
    "R": (0, 255, 0),  # Green for Residential
    "C": (0, 0, 255),  # Blue for Commercial
    "I": (255, 0, 0),  # Red for Industrial
    "G": (0, 255, 128),  # Light Green for Green Space
}
font = pygame.font.Font(None, 24)


def draw_statistics(city):
    """Draw statistics on the right side of the screen.

    Args:
        city (CitySimulation): The city simulation object containing metrics.
    """
    # Draw statistics on the right side of the screen
    stats_x = GRID_SIZE * CELL_SIZE + 10
    stats_y = 10
    # Create statistics text
    stats = [
        f"Generation: {city.generation_count}",
        f"Residential: {city.count_residential_cells()}",
        f"Commercial: {city.count_commercial_cells()}",
        f"Industrial: {city.count_industrial_cells()}",
        f"Green Space: {city.count_green_cells()}",
        f"Population Growth Rate: {city.population_growth_rate:.2f}",
        f"Environmental Impact Rate: {city.environmental_impact_rate:.2f}",
        f"Environmental Conservation Rate: {city.environmental_impact_rate:.2f}",
        f"Infrastructure Development Rate: {city.infrastructure_development_rate:.2f}",
    ]
    for i, stat in enumerate(stats):
        text_surface = font.render(stat, True, (255, 255, 255))  # White text
        screen.blit(text_surface, (stats_x, stats_y + i * 30))


def draw_grid(city):
    """Draw the grid representation of the city on the screen.

    Args:
        city (CitySimulation): The city simulation object to visualize.
    """
    for i in range(city.size):
        for j in range(city.size):
            cell_state = city.grid[i][j].state
            color = colors[cell_state]
            pygame.draw.rect(
                screen, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )


def draw_legend():
    """Draw the legend explaining the color coding of different cell types."""
    legend_x = 10  # Start drawing legend from the left
    legend_y = GRID_SIZE * CELL_SIZE + 10  # Position it below the grid
    legend_items = [
        ("Empty", "E", colors["E"]),
        ("Residential", "R", colors["R"]),
        ("Commercial", "C", colors["C"]),
        ("Industrial", "I", colors["I"]),
        ("Green Space", "G", colors["G"]),
    ]
    for i, (label, symbol, color) in enumerate(legend_items):
        pygame.draw.rect(
            screen, color, (legend_x, legend_y + i * 30, 20, 20)
        )  # Draw color square
        text_surface = font.render(
            f"{label} ({symbol})", True, (255, 255, 255)
        )  # White text
        screen.blit(
            text_surface, (legend_x + 30, legend_y + i * 30)
        )  # Position text next to square


class DQN:
    """Deep Q-Learning model for the agent.

    Attributes:
        state_dim (int): The dimension of the state space.
        action_dim (int): The dimension of the action space.
        memory (list): Memory for storing experiences.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration rate for epsilon-greedy action selection.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Decay rate for exploration.
        model (Sequential): The neural network model for Q-value prediction.
        target_model (Sequential): The target model for stability in training.
    """

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
        """Create and compile the neural network model.

        Returns:
            Sequential: The compiled Keras model.
        """
        model = Sequential()
        model.add(Dense(64, activation="relu", input_dim=self.state_dim))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_dim))
        model.compile(loss="mse", optimizer="adam")
        return model

    def update_target_model(self):
        """Update the target model with weights from the primary model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store the experience in memory.

        Args:
            state (np.array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.array): The next state after action.
            done (bool): Whether the episode is finished.
        """
        self.memory.append((state, action, reward, next_state, done))
        self.rewards.append(reward)
        self.actions_taken.append(action)

    def replay(self, batch_size):
        """Train the model using a batch of experiences from memory.

        Args:
            batch_size (int): Number of experiences to sample from memory.
        """
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state)[0]
                )
            target_f = self.model.predict(state)
            self.q_values.append(np.max(target_f))
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def act(self, state):
        """Choose an action based on the current state using epsilon-greedy policy.

        Args:
            state (np.array): The current state.

        Returns:
            int: The chosen action.
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def decay_epsilon(self):
        """Decay the exploration rate to reduce exploration over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class Agent:
    """Agent that interacts with the city simulation using a DQN.

    Attributes:
        city (CitySimulation): The city simulation object.
        dqn (DQN): The DQN model used by the agent.
        high_score_count (int): Count of high scores achieved.
    """

    def __init__(self, city, dqn):
        self.city = city
        self.dqn = dqn
        self.high_score_count = 0

    def act(self):
        """Choose an action based on the current state using epsilon-greedy policy.

        Args:
            state (np.array): The current state.

        Returns:
            int: The chosen action.
        """
        state = self.get_state()
        action = self.dqn.act(state)
        self.adapt_city(action)
        return action

    def get_state(self):
        """Get the current state representation of the city.

        Returns:
            np.array: The state as a 1D array.
        """
        return np.array(
            [
                self.city.count_empty_cells(),
                self.city.count_green_cells(),
                self.city.count_residential_cells(),
                self.city.count_industrial_cells(),
                self.city.count_commercial_cells(),
                self.city.population_growth_rate,
                self.city.environmental_impact_rate,
                self.city.infrastructure_development_rate,
                self.city.environmental_conservation_rate,
            ]
        ).reshape(1, -1)

    def adapt_city(self, action):
        """Adapt the city metrics based on the chosen action.

        Args:
            action (int): The action taken by the agent.
        """
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
        """Calculate the reward based on the current state of the city.

        Returns:
            float: The calculated reward.
        """
        city_value = self.city.sum_city_state_values()
        score = 100 - (abs(city_value - 50) / 50) * 100
        final_score = max(0, min(score, 100))
        if final_score > 90:
            self.high_score_count += 1
        return final_score


def plot_metrics(agent, dqn):
    """Plot the metrics (rewards and Q-values) over time.

    Args:
        agent (Agent): The agent interacting with the city.
        dqn (DQN): The DQN model used by the agent.
    """
    # Plot rewards over time
    plt.figure(figsize=(12, 6))

    rewards = np.array(dqn.rewards)
    success_count = np.sum(rewards > 80)
    total_attempts = len(rewards)
    success_rate = success_count / total_attempts if total_attempts > 0 else 0

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(dqn.rewards)
    plt.title("Rewards over Time")
    plt.xlabel("Generation")
    plt.ylabel("Reward")

    # Display success rate in the plot
    plt.text(
        0.5,
        0.9,
        f"Success Rate: {success_rate:.2%}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
        fontsize=10,
        color="black",
    )

    # Plot Q-values
    plt.subplot(1, 2, 2)
    plt.plot(dqn.q_values)
    plt.title("Q-values over Time")
    plt.xlabel("Generation")
    plt.ylabel("Q-value")

    plt.tight_layout()
    plt.show()
    plt.savefig("Q Values Over Time.png")  # Saves as a PNG file
    print("Plot saved as 'Q Values Over Time.png'")

    # Plot actions taken
    plt.figure(figsize=(12, 6))
    plt.hist(dqn.actions_taken, bins=dqn.action_dim, edgecolor="black")
    plt.title("Distribution of Actions Taken")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.show()
    plt.savefig("Distribution Of Actions Taken.png")  # Saves as a PNG file
    print("Plot saved as 'distribution_of_actions_taken.png'")


def plot_city(city):
    """Visualize the current state of the city.

    Args:
        city (CitySimulation): The city simulation object to visualize.
    """
    # Create a color map for different cell types
    color_map = {
        "E": [1, 1, 1],  # White for Empty
        "R": [0, 1, 0],  # Green for Residential
        "C": [0, 0, 1],  # Blue for Commercial
        "I": [1, 0, 0],  # Red for Industrial
        "G": [0, 1, 0.5],  # Light Green for Green Space
    }

    grid_colors = np.zeros((city.size, city.size, 3))
    for i in range(city.size):
        for j in range(city.size):
            cell_state = city.grid[i][j].state
            grid_colors[i, j] = color_map[cell_state]

    plt.imshow(grid_colors, interpolation="nearest")
    plt.title("City State Visualization")
    plt.axis("off")  # Hide the axis
    plt.show()


def main():
    """Main function to run the city evolution simulation.
    Initializes the city, agent, and starts the simulation loop.
    """
    city = CitySimulation(20, 1)
    dqn = DQN(9, 7)  # State has 9 dimensions, 7 actions
    agent = Agent(city, dqn)

    # Prepare for animation
    fig, ax = plt.subplots()
    img = ax.imshow(np.zeros((city.size, city.size, 3)), interpolation="nearest")
    plt.axis("off")  # Hide the axis
    frames = []  # Store the city states for animation
    clock = pygame.time.Clock()
    running = True
    for generation in range(1000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        state = agent.get_state()
        action = agent.act()
        reward = agent.calculate_reward()
        next_state = agent.get_state()
        print(f"Agent Score :{reward}")
        done = False

        dqn.remember(state, action, reward, next_state, done)
        dqn.replay(32)
        dqn.decay_epsilon()
        screen.fill((0, 0, 0))
        draw_grid(city)
        draw_statistics(city)
        draw_legend()
        pygame.display.flip()
        city.evolve()
        clock.tick(FPS)
        # Store the current state for animation
        frames.append(city)

    # ani = FuncAnimation(fig, update, frames=frames, fargs=(frames, img), interval=500)
    # plt.show()
    success_rate = np.sum(np.array(dqn.rewards) > 80) / len(dqn.rewards)
    # Calculate cumulative success rate
    cumulative_success_rate = np.cumsum(np.array(dqn.rewards) > 80) / np.arange(
        1, len(dqn.rewards) + 1
    )
    # Calculate average score above 80
    average_score_above_10 = np.mean(np.array(dqn.rewards)[np.array(dqn.rewards) > 80])
    # Calculate standard deviation of scores above 80
    std_dev_scores_above_10 = np.std(np.array(dqn.rewards)[np.array(dqn.rewards) > 80])
    plot_metrics(agent, dqn)


def update(city, frames, img):
    """Update function for animation.

    Args:
        city (CitySimulation): The current city simulation object.
        frames (list): The list of frames for the animation.
        img: The image object to be updated.

    Returns:
        img: The updated image object.
    """
    color_map = {
        "E": [1, 1, 1],  # White for Empty
        "R": [0, 1, 0],  # Green for Residential
        "C": [0, 0, 1],  # Blue for Commercial
        "I": [1, 0, 0],  # Red for Industrial
        "G": [0, 1, 0.5],  # Light Green for Green Space
    }

    grid_colors = np.zeros((frames.size, frames.size, 3))
    for i in range(frames.size):
        for j in range(frames.size):
            cell_state = frames.grid[i][j].state
            grid_colors[i, j] = color_map[cell_state]

    img.set_array(grid_colors)
    return (img,)


if __name__ == "__main__":
    main()
