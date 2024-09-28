import json

class InputModule:
    def receive_plan(self, plan_data):
        """
        Receive and validate the city plan input.
        
        Parameters:
            plan_data (dict): The city plan provided by the agents.
        
        Returns:
            dict: Validated city plan data.
        Raises:
            ValueError: If the plan format is invalid.
        """
        if self.validate_plan(plan_data):
            print("City plan received and validated successfully.")
            return plan_data
        else:
            raise ValueError("Invalid plan format. Ensure required fields are provided.")

    def validate_plan(self, plan_data):
        """
        Validate the structure of the city plan to ensure all necessary keys are present.
        
        Parameters:
            plan_data (dict): The city plan provided by the agents.
        
        Returns:
            bool: True if the plan is valid, False otherwise.
        """
        required_keys = ['name', 'zones', 'infrastructure', 'budget']
        if not all(key in plan_data for key in required_keys):
            print(f"Missing required keys: Expected {required_keys}")
            return False
        
        # Validate that 'zones' is a list of dictionaries
        if not isinstance(plan_data['zones'], list) or not all(isinstance(zone, dict) for zone in plan_data['zones']):
            print("Invalid format for zones. Expected a list of dictionaries.")
            return False
        
        return True
