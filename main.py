from critic.input_module import InputModule
from critic.evaluation_engine import EvaluationEngine
from critic.feedback_module import FeedbackGenerationModule
from critic.integration import IntegrationInterface

def run_critic_system():
    agent_id = "Agent_001"
    integration = IntegrationInterface()
    plan = integration.receive_plan_from_agent(agent_id)

    input_module = InputModule()
    validated_plan = input_module.receive_plan(plan)

    evaluation_engine = EvaluationEngine()
    scores = evaluation_engine.evaluate_plan(validated_plan)

    feedback_module = FeedbackGenerationModule()
    feedback = feedback_module.generate_feedback(scores)

    integration.send_feedback(agent_id, feedback)

if __name__ == "__main__":
    while True:
        run_critic_system()
