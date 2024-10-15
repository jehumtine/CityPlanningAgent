from agent import Agent
from city import CitySimulation
from critic import Critic
from learning import Learning


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

def test():
    city = CitySimulation(20, 0)
    agent = Agent(city, 0)
    agent.load_q_table()
    critic = Critic(agent)
    while True:
        critic.evaluate()


if __name__ == "__main__":
        train()
