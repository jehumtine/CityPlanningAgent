import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Learning:
    def __init__(self, learning_rate=0.1, exploration_rate=0.2):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate  # Probability of exploration
        self.q_table = {}  # Initialize Q-table

    def update(self, state, action, reward, next_state):
        # Basic Q-learning update rule
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0

        # Get the maximum Q-value for the next state
        best_next_action = max(self.q_table.get(next_state, {}), key=self.q_table.get(next_state, {}).get, default=0)

        # Update the Q-value using the learning rule
        self.q_table[state][action] += self.learning_rate * (reward + best_next_action - self.q_table[state][action])

    def select_action(self, agent):
        # Epsilon-greedy action selection/allows exploration
        if random.random() < self.exploration_rate:  # Explore
            return random.choice(["increase_growth", "reduce_impact", "improve_infrastructure", "decrease_pollution"])
        
        # State is the tuple of parameters from agent.city.city
        state = (agent.city.city['city_value'], 
                 agent.city.city['population_growth_rate'], 
                 agent.city.city['infrastructure_development_rate'], 
                 agent.city.city['environmental_conservation_rate'], 
                 agent.city.city['environmental_impact_rate'], 
                 agent.city.city['development_rate_mutation'])
        
        # Select the best action for a given state
        if state not in self.q_table or not self.q_table[state]:  # Default action if no Q-values exist
            return "default_action"
        
        return max(self.q_table[state], key=self.q_table[state].get)

    def adapt_city(self, agent):
        # State is directly extracted from agent.city.city
        state = (agent.city.city['city_value'], 
                 agent.city.city['population_growth_rate'], 
                 agent.city.city['infrastructure_development_rate'], 
                 agent.city.city['environmental_conservation_rate'], 
                 agent.city.city['environmental_impact_rate'], 
                 agent.city.city['development_rate_mutation'])
        
        action = self.select_action(agent)  # Action based on the current state
        reward = self.calculate_reward(agent)  # Get the score from the critic

        # Adjust the city parameters based on the selected action
        if action == "increase_growth":
            agent.city.city['population_growth_rate'] += 0.01  # Example adjustment
        elif action == "reduce_impact":
            agent.city.city['environmental_impact_rate'] -= 0.01  # Example adjustment
        elif action == "improve_infrastructure":
            agent.city.city['infrastructure_development_rate'] += 0.01  # Example adjustment
        elif action == "decrease_pollution":
            agent.city.city['environmental_impact_rate'] -= 0.02  # Stronger adjustment

        # Update Q-table with the current state, action taken, reward received, and the new state
        next_state = (agent.city.city['city_value'], 
                      agent.city.city['population_growth_rate'], 
                      agent.city.city['infrastructure_development_rate'], 
                      agent.city.city['environmental_conservation_rate'], 
                      agent.city.city['environmental_impact_rate'], 
                      agent.city.city['development_rate_mutation'])
        
        self.update(state, action, reward, next_state)

        # Print all the parameters to the console
        self.print_parameters(agent.city.city)

        # Log the action taken
        logging.info(f'Action taken: {action}, New State: {next_state}, Reward: {reward}')


    def print_parameters(self, city):
        # Create a dictionary of the city parameters to print them easily
        output = {
            "City Value": city['city_value'],
            "Population Growth Rate": city['population_growth_rate'],
            "Infrastructure Development Rate": city['infrastructure_development_rate'],
            "Environmental Conservation Rate": city['environmental_conservation_rate'],
            "Environmental Impact Rate": city['environmental_impact_rate'],
            "Development Rate Mutation": city['development_rate_mutation']
        }

        # Print each parameter and its current value
        for key, value in output.items():
            print(f"{key}: {value}")
            
    def calculate_reward(self, agent):
        # Implement reward calculation (placeholder logic)
        return agent.score  # use the score from the critic as the reward

    def select_new_parameter(self, current_value):
        # Determine a new parameter value based on Q-learning or other strategy
        return current_value + (self.learning_rate * (1 - current_value))
