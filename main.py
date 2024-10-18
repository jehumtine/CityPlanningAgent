from agent import Agent
from city import CitySimulation
from critic import Critic
from learning import Learning, CityState


def train():
    city = CitySimulation(20, 1)
    agent = Agent(city, 0)
    critic = Critic(agent)
    learning = Learning(agent=agent)
    while True:
        critic.evaluate()
        learning.load_q_table()
        learning.adapt_city(agent)

        learning.save_q_table()

def test_agent():
    city = CitySimulation(20, 0)
    agent = Agent(city, 0)
    learning = Learning(agent)

    # Load the Q-table from the file
    learning.load_q_table("q_table.txt")

    # Test the agent for a certain number of generations
    for generation in range(1000):
        # Get the current state of the city
        current_state = CityState(
            agent.city.count_empty_cells(),
            agent.city.count_green_cells(),
            agent.city.count_residential_cells(),
            agent.city.count_industrial_cells(),
            agent.city.count_commercial_cells(),
            agent.city.population_growth_rate,
            agent.city.environmental_impact_rate,
            agent.city.infrastructure_development_rate,
            agent.city.environmental_conservation_rate
        )

        # Select an action based on the Q-table
        action = learning.select_action(current_state)
        # Update the city state based on the action
        if action == "increase_growth":
            agent.city.population_growth_rate += 0.08
        elif action == "reduce_impact":
            agent.city.environmental_impact_rate -= 0.1
        elif action == "improve_infrastructure":
            agent.city.infrastructure_development_rate += 0.08
        elif action == "decrease_pollution":
            agent.city.environmental_impact_rate -= 0.06
        elif action == "increase_housing_capacity":
            agent.city.population_growth_rate += 0.08
        elif action == "promote_commercial_growth":
            agent.city.infrastructure_development_rate += 0.1
        elif action == "stabilize_growth":
            agent.city.population_growth_rate += 0.09
            agent.city.infrastructure_development_rate += 0.8
            agent.city.environmental_impact_rate += 0.06
        critic = Critic(agent)
        critic.evaluate()
        # Take the action and get the reward
        reward = learning.calculate_reward(agent)



        # Print the current state and action
        print(f"Generation: {generation+1}")
        print(f"Current State: {current_state}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print()

        # Evolve the city
        agent.city.evolve()



if __name__ == "__main__":
        train()
