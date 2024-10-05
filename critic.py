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
        print("Evolving the city based on agent's actions...")
        self.agent.city.evolve()  # Accessing the city's evolve method
        
        # Calculate the city score after evolution
        city_score = self._calculate_city_score(self.agent.city)
        print(f"City score after evolution: {city_score}")
        
        # Update the agent's score based on the city score
        self.agent.score = city_score
        print(f"Updated agent's score: {self.agent.score}")

    def _calculate_city_score(self, city):
        """
        Calculate a score for the city based on its state after evolution.
        Args:
            city (CitySimulation): The city instance.
        Returns:
            float: A score for the city.
        """
        # Example scoring logic based on city parameters
        score = 0
        score += city.population_growth_rate * 20
        score += city.infrastructure_development_rate * 30
        score -= city.environmental_impact_rate * 25
        score += city.environmental_conservation_rate * 25

        # Normalize the score to a scale of 0-100
        score = max(0, min(score, 100))
        return score
