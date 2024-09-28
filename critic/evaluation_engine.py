import numpy as np

from city import CitySimulation


# If you use NumPy for calculations


class EvaluationEngine:
    def evaluate_plan(self, plan):
        """
        Evaluate the entire city plan across multiple dimensions.
        
        Parameters:
            plan (dict): The validated city plan to be evaluated.
        
        Returns:
            dict: A dictionary containing evaluation scores.
        """
        city = CitySimulation(40, cityRule)
        for i in range(9):
            city.evolve()
        infrastructuredevrate = city.infrastructure_development_rate




        efficiency_scores = self.evaluate_efficiency(plan)
        functionality_scores = self.evaluate_functionality(plan)
        aesthetics_scores = self.evaluate_aesthetics(plan)
        sustainability_scores = self.evaluate_sustainability(plan)
        
        return {
            'efficiency': efficiency_scores,
            'functionality': functionality_scores,
            'aesthetics': aesthetics_scores,
            'sustainability': sustainability_scores
        }

    # Efficiency Evaluation
    def evaluate_efficiency(self, plan):
        """
        Evaluate the efficiency of land use, budget allocation, and infrastructure accessibility.
        
        Parameters:
            plan (dict): The city plan containing zones and budget information.
        
        Returns:
            dict: Efficiency scores for land use, budget, and infrastructure.
        """
        density_score = self.calculate_density(plan)
        budget_efficiency_score = self.assess_budget_efficiency(plan)
        infrastructure_accessibility_score = self.evaluate_infrastructure_accessibility(plan)

        return {
            'density': density_score,
            'budget_efficiency': budget_efficiency_score,
            'infrastructure_accessibility': infrastructure_accessibility_score
        }

    def calculate_density(self, plan):
        """Calculate land use density as a ratio of residential land to total land."""
        total_land = sum(zone['area'] for zone in plan['zones'])
        residential_land = sum(zone['area'] for zone in plan['zones'] if zone['type'] == 'residential')
        return residential_land / total_land if total_land > 0 else 0

    def assess_budget_efficiency(self, plan):
        """Assess budget efficiency by comparing actual vs allocated budget."""
        return plan['budget']['actual'] / plan['budget']['allocated'] if plan['budget']['allocated'] > 0 else 0

    def evaluate_infrastructure_accessibility(self, plan):
        """Calculate average infrastructure accessibility score across zones."""
        accessibility_scores = [zone.get('accessibility', 0) for zone in plan['zones']]
        return sum(accessibility_scores) / len(accessibility_scores) if accessibility_scores else 0

    # Functionality Evaluation
    def evaluate_functionality(self, plan):
        """Evaluate zoning balance, service coverage, and integration."""
        zoning_balance_score = self.check_zoning_balance(plan)
        service_coverage_score = self.evaluate_service_coverage(plan)
        functional_integration_score = self.measure_functional_integration(plan)

        return {
            'zoning_balance': zoning_balance_score,
            'service_coverage': service_coverage_score,
            'functional_integration': functional_integration_score
        }

    def check_zoning_balance(self, plan):
        """Evaluate how balanced the distribution of zoning types is."""
        total_area = sum(zone['area'] for zone in plan['zones'])
        zoning_areas = {zone['type']: sum(z['area'] for z in plan['zones'] if z['type'] == zone['type']) for zone in plan['zones']}
        return {zone: area / total_area for zone, area in zoning_areas.items()}

    def evaluate_service_coverage(self, plan):
        """Evaluate how well services are distributed and accessible."""
        return sum(1 for zone in plan['zones'] if zone.get('accessibility', 0) > 0.5) / len(plan['zones'])

    def measure_functional_integration(self, plan):
        """Measure how well land use integrates with existing infrastructure."""
        integration_score = sum(zone.get('accessibility', 0) for zone in plan['zones']) / len(plan['zones'])
        return integration_score

    # Aesthetic Evaluation
    def evaluate_aesthetics(self, plan):
        """Evaluate the visual appeal of the city."""
        aesthetics_score = sum(7 if zone['type'] == 'residential' else 5 if zone['type'] == 'commercial' else 3 for zone in plan['zones'])
        return aesthetics_score / len(plan['zones']) if plan['zones'] else 0

    # Sustainability Evaluation
    def evaluate_sustainability(self, plan):
        """Evaluate the sustainability of the city plan."""
        environmental_impact_score = self.evaluate_environmental_impact(plan)
        energy_efficiency_score = self.calculate_energy_efficiency(plan)

        return {
            'environmental_impact': environmental_impact_score,
            'energy_efficiency': energy_efficiency_score
        }

    def evaluate_environmental_impact(self, plan):
        """Assess the environmental impact based on zoning and area usage."""
        environmental_score = sum(zone['area'] * (1 - zone.get('accessibility', 0)) for zone in plan['zones'])
        return environmental_score / sum(zone['area'] for zone in plan['zones']) if plan['zones'] else 0

    def calculate_energy_efficiency(self, plan):
        """Evaluate energy efficiency based on infrastructure and zone types."""
        energy_score = sum(zone['area'] * (1 / zone.get('accessibility', 1)) for zone in plan['zones'])
        return energy_score / sum(zone['area'] for zone in plan['zones']) if plan['zones'] else 0
