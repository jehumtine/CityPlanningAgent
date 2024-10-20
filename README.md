# City Evolution Simulation

## Overview

This project implements a simple AI agent designed to manage a city simulation using reinforcement learning techniques. The agent interacts with a grid that represents various city zones (residential, commercial, industrial, green spaces) and adapts its strategies to optimize city growth while minimizing environmental impact.

## Objectives

- Understand and apply AI foundations in a practical scenario.
- Design and implement a reinforcement learning agent using TensorFlow and Pygame for visualization.

## Features

- Visual representation of city evolution with color-coded zones.
- An AI agent utilizing Deep Q-Learning to make decisions based on city metrics.
- Real-time statistics display for city metrics.

## Getting Started

### Prerequisites

- Python 3.x
- Required Libraries:
  - NumPy
  - TensorFlow
  - Pygame
  - Matplotlib

You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

### Running the Simulation

Clone the repository:

```bash
git clone https://github.com/yourusername/city-evolution-simulation.git
cd city-evolution-simulation
```

Run the main script:

```bash
python main.py
```

In the terminal you can select between training and testing the model to use for the simulation.

```bash
Enter mode (train/test): train
```

From then the output of the simulation will be shown in a pygame window.
In test mode, the output can be paused and observed at any time by clicking the space button.

## Project Structure

```bash
city-evolution-simulation/
├── main.py              # Main simulation code
├── city.py              # City simulation logic
├── learning.py          # DQN and learning implementation
├── critic.py            # Critic evaluation logic for agent scoring
├── agent.py             # Agent implementation logic
└── README.md            # Project documentation
```

## Contributions

This project was developed collaboratively by a group of students, with each member contributing to different aspects of the design and implementation:

- **Jehu:** *city.py* - Developed the city implementation and representation.
- **Ziba:** *learning.py* - Implemented the DQN and learning algorithms for the project.
- **Twange:** *critic.py* - Created the critic evaluation logic used for agent score calculation.
- **Farai:** *main.py* - Designed the user interface and visualization of the simulation using Pygame.
- **Kangwa:** *agent.py* - Implemented the core logic for the AI agent.
