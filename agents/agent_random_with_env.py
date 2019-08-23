from rl_gym import envs
from collections import OrderedDict
from time import time
import redis
import json
from numpy.random import random_integers, random_sample, shuffle
from time import sleep


class AgentDesc:
    def __init__(self):
        # print('here1')
        # self.agents = ["agent1", "agent2", "agent3"]
        self.agents = ['agent{}'.format(i + 1) for i in range(25)]
        self.env = envs.fin_env_pol_gra.FinEnvTest(self.agents)
        self.assets = self.env.assets
        self.data_processed = []

        self.index = 0
        with open('rl_gym/envs/data.json') as f:
            data_loaded = json.load(f)
            for val in data_loaded:
                del val['_id']
                val['sequence'] = val['sequence']['$numberLong']
                for bid in val['bids']:
                    del bid[2]
                for ask in val['asks']:
                    del ask[2]
                self.data_processed.append(val)
            self.data_processed = sorted(self.data_processed, key=lambda x: x['sequence'])

    def run_agent(self):
        # bid_array = [float(self.data_processed[self.index]["bids"][i][0]) for i in range(900)]
        # ask_array = [float(self.data_processed[self.index]["asks"][i][0]) for i in range(900)]
        agents = ['agent{}'.format(i + 1) for i in range(0, 20)]
        for j in range(900):
            shuffle(agents)
            actions = OrderedDict()

            buy_agent = agents[0:10]
            for i in range(len(buy_agent)):
                actions[buy_agent[i]] = OrderedDict()
                for asset in self.assets:
                    actions[buy_agent[i]][asset] = OrderedDict()
                    actions[buy_agent[i]][asset]["action"] = 2
                    actions[buy_agent[i]][asset]["limit_max"] = float(self.data_processed[self.index]["bids"][i][0])
                    actions[buy_agent[i]][asset]["volume"] = float(self.data_processed[self.index]["bids"][i][1])
                    actions[buy_agent[i]][asset]["time"] = time()
                    actions[buy_agent[i]][asset]["order_type"] = "limit"

            sell_agents = agents[-10:]
            sell_index = 0

            for sell_agent in sell_agents:
                actions[sell_agent] = OrderedDict()
                for asset in self.assets:
                    actions[sell_agent][asset] = OrderedDict()
                    actions[sell_agent][asset]["action"] = 1
                    actions[sell_agent][asset]["limit_min"] = float(
                        self.data_processed[self.index]["asks"][sell_index][0])
                    actions[sell_agent][asset]["volume"] = float(self.data_processed[self.index]["asks"][sell_index][1])
                    actions[sell_agent][asset]["time"] = time()
                    actions[sell_agent][asset]["order_type"] = "limit"
                sell_index += 1

            rand_agents = ['agent{}'.format(i) for i in range(21, 26)]

            for rand_agent in rand_agents:
                actions[rand_agent] = OrderedDict()
                for asset in self.assets:
                    actions[rand_agent][asset] = OrderedDict()
                    actions[rand_agent][asset]["action"] = self.env.asset_action_space.sample()
                    if actions[rand_agent][asset]["action"] == 1:
                        if self.env.agent_holdings[rand_agent][asset] < 1:
                            actions[rand_agent][asset]["action"] = 0
                        else:
                            actions[rand_agent][asset]["volume"] = float(
                                random_integers(1, self.env.agent_holdings[rand_agent][asset]))
                    elif actions[rand_agent][asset]["action"] == 2:
                        actions[rand_agent][asset]["volume"] = float(random_integers(1, 2))
                    actions[rand_agent][asset]["limit_min"] = float(
                        random_integers(0.950 * self.env.last_traded_price[asset],
                                        0.990 * self.env.last_traded_price[asset]))
                    actions[rand_agent][asset]["limit_max"] = float(
                        random_integers(1.01 * self.env.last_traded_price[asset],
                                        1.05 * self.env.last_traded_price[asset]))
                    actions[rand_agent][asset]["order_type"] = "limit"
                    actions[rand_agent][asset]["time"] = time()

            self.index += 1
            rew = self.env.step(actions)
            # print(self.env.infer("asset_0", "reward"))
            sleep(0.5)
            # print(self.env.last_traded_price['asset_0'])


if __name__ == '__main__':
    agent = AgentDesc()
    agent.run_agent()
