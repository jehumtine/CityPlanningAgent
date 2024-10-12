import random
import logging
from collections import namedtuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a named tuple for city state representation
CityState = namedtuple('CityState', ['population_growth_rate', 'environmental_impact_rate', 'infrastructure_development_rate'])

class Learning:
    def __init__(self,agent ,learning_rate=0.1, exploration_rate=0.2):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate  # Probability of exploration
        self.q_table = {}  # Initialize Q-table
        self.agent = agent

    def update(self, state, action, reward, next_state):
        # Basic Q-learning update rule
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0

        # Get the maximum Q-value for the next state
        best_next_action = max(self.q_table.get(next_state, {}).values(), default=0)

        # Update the Q-value using the learning rule
        self.q_table[state][action] += self.learning_rate * (reward + best_next_action - self.q_table[state][action])

    def select_action(self, state):
        # Epsilon-greedy action selection/allows exploration
        if random.random() < self.exploration_rate:  # Explore
            return random.choice(["increase_growth", "reduce_impact", "improve_infrastructure", "decrease_pollution"])
        
        # Select the best action for a given state
        if state not in self.q_table or not self.q_table[state]:  # Don't change this line
            return "default_action"  # Default action if no Q-values exist
        
        return max(self.q_table[state], key=self.q_table[state].get)

    def adapt_city(self, agent):
        # Adapt the city's parameters based on the agent's score and current state
        current_state = CityState(
            agent.city.population_growth_rate,
            agent.city.environmental_impact_rate,
            agent.city.infrastructure_development_rate  # Assuming infrastructure development rate may not be initialized
        )
        
        action = self.select_action(current_state)  # Action based on the parameters pushed
        reward = self.calculate_reward(agent)  # Get the score from the critic

        # Logic to adjust the city parameters based on the selected action
        if action == "increase_growth":
            agent.city.population_growth_rate += 0.01  # adjustment
        elif action == "reduce_impact":
            agent.city.environmental_impact_rate -= 0.01  #  adjustment
        elif action == "improve_infrastructure":
            # adjustment
            agent.city.infrastructure_development_rate += 0.01
        elif action == "decrease_pollution":
            #  adjustment
            agent.city.environmental_impact_rate -= 0.02  #  adjustment
        
        # Update Q-table with the current state, action taken, reward received, and the new state
        next_state = CityState(
            agent.city.population_growth_rate,
            agent.city.environmental_impact_rate,
            agent.city.infrastructure_development_rate
        )
        self.update(current_state, action, reward, next_state)
        # Log the action taken
        logging.info(f'Action taken: {action}, New State: {next_state}, Reward: {reward}, Q-Table; {self.q_table.items()}')

    def calculate_reward(self, agent) -> float:
        # Implement a nuanced reward function based on agent's performance and city state
        # This is a placeholder example; adjust as needed
        return agent.score  # For now, use the score from the critic as the reward

    def select_new_parameter(self, current_value):
        # Logic to determine a new parameter value based on Q-learning or another strategy
        return current_value + (self.learning_rate * (1 - current_value))  # Example update


