import numpy as np
import tensorflow as tf

from city import CitySimulation
from agent import Agent
from learning import DQN

import matplotlib.pyplot as plt

from city import CitySimulation
import pygame

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
    "R": (255, 165, 0),  # Green for Residential
    "C": (0, 0, 255),  # Blue for Commercial
    "I": (255, 0, 0),  # Red for Industrial
    "G": (0, 255, 128),  # Light Green for Green Space
}
font = pygame.font.Font(None, 24)
TRAINING_MODE = "train"
TESTING_MODE = "test"


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


def plot_metrics(agent, dqn):
    """Plot the metrics (rewards and Q-values) over time.

    Args:
        agent (Agent): The agent interacting with the city.
        dqn (DQN): The DQN model used by the agent.
    """
    # Plot rewards over time
    plt.figure(figsize=(12, 6))

    rewards = np.array(dqn.rewards)
    success_count = np.sum(rewards > 70)
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
    plt.savefig("./images/Rewards Over Time.png")  # Saves as a PNG file
    plt.show()

    # Plot actions taken
    plt.figure(figsize=(12, 6))
    plt.hist(dqn.actions_taken, bins=dqn.action_dim, edgecolor="black")
    plt.title("Distribution of Actions Taken")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.savefig("./images/Distribution Of Actions Taken.png")  # Saves as a PNG file
    plt.show()
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


def main(mode=TRAINING_MODE):
    """Main function to run the city evolution simulation.
    Initializes the city, agent, and starts the simulation loop.
    """
    city = CitySimulation(20, 1)
    dqn = DQN(9, 7)  # State has 9 dimensions, 7 actions
    agent = Agent(city, dqn)
    paused = False
    model_path = "./training_model/city_planning_model.keras"
    if mode == TRAINING_MODE:
        clock = pygame.time.Clock()
        running = True
        # Number of generations to simulate
        generations = 100
        # Number of q_table values to replay from memory
        replay_value = 3
        for _ in range(generations):
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
            dqn.replay(replay_value)
            dqn.decay_epsilon()
            screen.fill((0, 0, 0))
            draw_grid(city)
            draw_statistics(city)
            draw_legend()
            pygame.display.flip()
            city.evolve()
            clock.tick(FPS)
        dqn.model.save(model_path)  # Save the trained model
        plot_metrics(agent, dqn)

    elif mode == TESTING_MODE:
        # Testing Logic
        dqn.model = tf.keras.models.load_model(model_path)
        # Run the testing Loop
        running = True
        clock = pygame.time.Clock()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
            if not paused:
                state = agent.get_state()
                action = agent.act()  # Use the model to predict actions
                reward = agent.calculate_reward()
                dqn.rewards.append(reward)
                dqn.actions_taken.append(action)
                print(f"Action taken: {action}")
                print(f"Agent Reward: {reward}")
                # Update the city state and display
                screen.fill((0, 0, 0))
                draw_grid(city)
                draw_statistics(city)
                draw_legend()
                pygame.display.flip()
                city.evolve()
                clock.tick(FPS)
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
    mode = input("Enter mode (train/test): ").strip().lower()
    if mode == "train":
        main(mode=TRAINING_MODE)
    elif mode == "test":
        main(mode=TESTING_MODE)
    else:
        print("Invalid mode. Please enter 'train' or 'test'.")
