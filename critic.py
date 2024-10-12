class Critic:
    def __init__(self, agent):
        """
        Initialize the Critic with the agent.
        Args:
            agent (Agent): The agent whose actions will be critiqued.
        """
        self.agent = agent

    def evaluate(self):
        """
        Evaluate the Agent's actions by evolving the city and scoring the results.
        """
        # Evolve the city through the agent
        print("Evolving the city based on agent's actions for 1000 generations...")
        for i in range(1000):
            self.agent.city.evolve()  # Accessing the city's evolve method

        
        # Calculate the city score after evolution
        city_score = self._calculate_city_score(self.agent.city)
        print(f"Agent score after evolution: {city_score}")
        
        # Update the agent's score based on the city score
        self.agent.score = city_score
        print(f"Updated agent's score")

    def _calculate_city_score(self, city):
        """
        Calculate a score for the city based on how close the city value is to 50.
        Args:
            city (CitySimulation): The city instance.
        Returns:
            float: A score for the city, where 50 is the optimal value.
        """
        # Get the city value (assuming it's a single value representing the city's state)
        city_value = city.sum_city_state_values()

        # Calculate the score based on how close city_value is to 50
        score = 100 - (abs(city_value - 50) / 50) * 100

        # Normalize the score to a range of 0-100
        score = max(0, min(score, 100))

        return score

