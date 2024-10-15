import logging
import json

from learning import CityState


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
        self.test = False
        self.q_table = {}


    def load_q_table(self, file_name='q_table.txt'):
        """Load the Q-table from a text file, converting strings back to CityState."""
        try:
            with open(file_name, 'r') as file:
                serializable_q_table = json.load(file)

            # Convert the strings back to CityState objects
            self.q_table = {
                self.string_to_city_state(state_str): actions for state_str, actions in serializable_q_table.items()
            }
            logging.info(f'Q-table loaded from {file_name}')
        except FileNotFoundError:
            logging.warning(f'File {file_name} not found. Starting with an empty Q-table.')

    def string_to_city_state(self, state_str):
        """Convert a string representation of CityState back to a CityState object."""
        # Assuming the string is in the format "CityState(population_growth_rate=0.02, environmental_impact_rate=0.03, ...)"
        # Extract the values from the string and convert them back to float
        values = state_str.replace("CityState(", "").replace(")", "").split(", ")
        values = [float(v.split('=')[1]) for v in values]

        return CityState(*values)


    def tweak_city(self):
        """
        Modify the city configuration.
        This is where the agent can make changes to the city (e.g., adjust growth rates).
        """
        # Example: Adjust the city's development rate
        self.city.infrastructure_development_rate += 0.1
        print(f"City's development rate increased to {self.city.infrastructure_development_rate}")
