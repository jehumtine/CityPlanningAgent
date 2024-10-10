class Agent:
    def __init__(self, city, score=0):
        """
        Initialize the Agent with a city and a score.
        Args:
            city (CitySimulation): The city the agent will work with.
            score (float): The agent's score. Defaults to 0.
        """
        self.city = city
        self.score = score

    def tweak_city(self):
        """
        Modify the city configuration.
        This is where the agent can make changes to the city (e.g., adjust growth rates).
        """
        # Example: Adjust the city's development rate
        self.city.infrastructure_development_rate += 0.1
        print(f"City's development rate increased to {self.city.infrastructure_development_rate}")
