import numpy as np


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
