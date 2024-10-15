import time

from agent import Agent
from city import CitySimulation
from critic import Critic
from learning import Learning


def run():
    city = CitySimulation(20, 0)
    agent = Agent(city, 0)
    critic = Critic(agent)
    learning = Learning(agent=agent)
    while True:
        critic.evaluate()
        learning.load_q_table()
        learning.adapt_city(agent)
        agent.city = CitySimulation(20,1)
        learning.save_q_table()

if __name__ == "__main__":
        run()
