import time

from agent import Agent
from city import CitySimulation
from critic import Critic
from learning import Learning


def run():
    city = CitySimulation(20, 1, True)
    agent = Agent(city, 0)
    critic = Critic(agent)
    learning = Learning(agent=agent)
    while True:

        critic.evaluate()
        learning.adapt_city(agent)
        time.sleep(2)
        agent.city = CitySimulation(20,1 , True)

if __name__ == "__main__":
        run()
