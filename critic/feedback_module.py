class FeedbackGenerationModule:
    def generate_feedback(self, scores):
        """
        Generate feedback based on evaluation scores.
        
        Parameters:
            scores (dict): A dictionary of evaluation scores across multiple criteria.
        
        Returns:
            dict: A dictionary containing feedback, suggestions, and rewards/penalties.
        """
        feedback = {
            'scores': scores,
            'suggestions': self.suggest_improvements(scores),
            'reward': self.assign_rewards(scores),
            'penalty': self.assign_penalties(scores)
        }
        return feedback

    def suggest_improvements(self, scores):
        """
        Generate specific improvement suggestions based on evaluation scores.
        
        Parameters:
            scores (dict): A dictionary of evaluation scores across multiple criteria.
        
        Returns:
            list: A list of suggestions for improving the city plan.
        """
        suggestions = []
        if scores['efficiency']['budget_efficiency'] < 0.8:
            suggestions.append("Optimize budget allocation for better cost-efficiency.")
        if scores['functionality']['service_coverage'] < 0.75:
            suggestions.append("Increase service coverage in under-served areas.")
        if 'aesthetics' in scores and scores['aesthetics'] < 5:
            suggestions.append("Enhance green spaces to improve visual appeal.")
        if scores['sustainability']['environmental_impact'] > 7:
            suggestions.append("Reduce environmental impact by improving energy efficiency.")
        
        return suggestions

    def assign_rewards(self, scores):
        """Assign rewards based on high performance in evaluation scores."""
        if scores['efficiency']['budget_efficiency'] >= 0.9 and scores['sustainability']['energy_efficiency'] > 0.8:
            return 100  # Example reward value
        return 0

    def assign_penalties(self, scores):
        """Assign penalties for poor performance in evaluation scores."""
        if scores['efficiency']['budget_efficiency'] < 0.5 or scores['sustainability']['environmental_impact'] > 8:
            return -50  # Example penalty value
        return 0
