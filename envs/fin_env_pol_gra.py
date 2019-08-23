import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections import OrderedDict
from time import time
import json
import numpy as np
import pandas as pd
# from copy import deepcopy
from operator import itemgetter
import redis
from copy import deepcopy


class FinEnvPolicyGradientTest(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, agents):
        self.assets = ['asset_{}'.format(i) for i in range(1)]
        self.last_traded_price = OrderedDict()
        # self.price_history = OrderedDict()
        self.buy_orders = OrderedDict()
        self.sell_orders = OrderedDict()
        self.shares_issued = OrderedDict()
        self.asset_action_space = spaces.Discrete(3)
        self.agents = agents
        self.agent_history = OrderedDict()
        self.agent_holdings = OrderedDict()
        self.state = OrderedDict()
        self.trades_processed = OrderedDict()
        # self.buy_sell_matches = OrderedDict()
        # self.simple_buy_orders = OrderedDict()
        # self.simple_sell_orders = OrderedDict()
        self.agent_actions = OrderedDict()
        self.transaction_cost = 6 / 10000.0
        self.trade_num = 0
        self.rewards = OrderedDict()
        self.market_buy_orders = OrderedDict()
        self.market_sell_orders = OrderedDict()

        for agent in self.agents:
            self.agent_history[agent] = OrderedDict()
            self.agent_holdings[agent] = OrderedDict()
            self.agent_actions[agent] = OrderedDict()
            self.state[agent] = {}
            self.rewards[agent] = OrderedDict()

        for asset in self.assets:
            self.last_traded_price[asset] = 600
            self.buy_orders[asset] = []
            self.sell_orders[asset] = []
            self.market_buy_orders[asset] = []
            self.market_sell_orders[asset] = []
            self.shares_issued[asset] = 1000
            # self.price_history[asset] = []
            self.trades_processed[asset] = []
            # self.buy_sell_matches[asset] = OrderedDict()
            # self.simple_buy_orders[asset] = []
            # self.simple_sell_orders[asset] = []
            for agent in self.agents:
                self.agent_actions[agent][asset] = []
                self.agent_history[agent][asset] = []
                self.agent_holdings[agent][asset] = 100
                # self.agent_holdings[agent][asset] = [200]
                self.rewards[agent][asset] = 0.0

    def step(self, actions):
        agent_num = 1

        for agent in self.agents:
            for asset in self.assets:
                action = actions[agent][asset]
                self.agent_actions[agent][asset].append(action)

                # Market order
                if action['order_type'] == 'market':

                    # Hold action
                    if action['action'] == 0:
                        pass

                    # Sell action
                    elif action['action'] == 1:

                        # Check if buy orders are present
                        if self.buy_orders[asset]:
                            buy_orders = sorted(self.buy_orders[asset],
                                                key=itemgetter('limit_max', 'order_time'))

                            # Loop through all buy orders to satisfy the order
                            for order in buy_orders:

                                # If action volume to sell is less than order volume to buy
                                if action['volume'] < order['volume']:

                                    # Reduce the action volume from order volume
                                    order['volume'] = order['volume'] - action['volume']

                                    # Append the trade to the trades processed list
                                    self.trades_processed[asset].append(
                                        {'buy_agent': order['agent'], 'volume': action['volume'],
                                         'price': order['limit_max'],
                                         'sell_agent': agent, 'transaction_time': time()})

                                    # Update the last traded price
                                    self.last_traded_price[asset] = order['limit_max']

                                    # Update agent history and holdings of buyer
                                    self.agent_history[order['agent']][asset].append(
                                        {'transaction_type': 'buy', 'volume': action['volume'],
                                         'price': order['limit_max'], 'transaction_time': time()})
                                    self.agent_holdings[order['agent']][asset] += action['volume']

                                    # Update agent history and holdings of seller
                                    self.agent_history[agent][asset].append(
                                        {'transaction_type': 'sell', 'volume': action['volume'],
                                         'price': order['limit_max'], 'transaction_time': time()})
                                    self.agent_holdings[agent][asset] -= action['volume']

                                    # Make action volume to 0
                                    action['volume'] = 0
                                    break

                                # If action volume to sell is greater than order volume to buy
                                elif action['volume'] > order['volume']:

                                    # Reduce the order volume from action volume
                                    action['volume'] = action['volume'] - order['volume']

                                    # Append the trade to the trades processed list
                                    self.trades_processed[asset].append(
                                        {'buy_agent': order['agent'], 'volume': order['volume'],
                                         'price': order['limit_max'],
                                         'sell_agent': agent, 'transaction_time': time()})

                                    # Update the last traded price
                                    self.last_traded_price[asset] = order['limit_max']

                                    # Remove the order from the buy orders list
                                    self.buy_orders[asset].remove(order)

                                    # Update agent history and holdings of buyer
                                    self.agent_history[order['agent']][asset].append(
                                        {'transaction_type': 'buy', 'volume': order['volume'],
                                         'price': order['limit_max'], 'transaction_time': time()})
                                    self.agent_holdings[order['agent']][asset] += order['volume']

                                    # Update agent history and holdings of seller
                                    self.agent_history[agent][asset].append(
                                        {'transaction_type': 'sell', 'volume': order['volume'],
                                         'price': order['limit_max'], 'transaction_time': time()})
                                    self.agent_holdings[agent][asset] -= order['volume']

                                # If action volume to sell is equal to order volume to buy
                                elif action['volume'] == order['volume']:

                                    # Append the trade to the trades processed list
                                    self.trades_processed[asset].append(
                                        {'buy_agent': order['agent'], 'volume': order['volume'],
                                         'price': order['limit_max'],
                                         'sell_agent': agent, 'transaction_time': time()})

                                    # Update the last traded price
                                    self.last_traded_price[asset] = order['limit_max']

                                    # Remove the order from the buy orders list
                                    self.buy_orders[asset].remove(order)

                                    # Update agent history and holdings of buyer
                                    self.agent_history[order['agent']][asset].append(
                                        {'transaction_type': 'buy', 'volume': order['volume'],
                                         'price': order['limit_max'], 'transaction_time': time()})
                                    self.agent_holdings[order['agent']][asset] += order['volume']

                                    # Update agent history and holdings of seller
                                    self.agent_history[agent][asset].append(
                                        {'transaction_type': 'sell', 'volume': order['volume'],
                                         'price': order['limit_max'], 'transaction_time': time()})
                                    self.agent_holdings[agent][asset] -= order['volume']

                                    # Make action volume to 0
                                    action['volume'] = 0
                                    break

                            # If limit sell orders get exhausted append to market sell orders list
                            if action['volume'] > 0:
                                self.market_sell_orders[asset].append(
                                    {'agent': agent, 'order_time': time(), 'volume': action['volume']})

                        # If no limit buy orders are present append to market sell orders list
                        else:
                            self.market_sell_orders[asset].append(
                                {'agent': agent, 'order_time': time(), 'volume': action['volume']})

                    # Buy action
                    elif action['action'] == 2:

                        # Check if sell orders are present
                        if self.sell_orders[asset]:
                            sell_orders = sorted(self.sell_orders[asset],
                                                 key=itemgetter('limit_min', 'order_time'))

                            # Loop through all sell orders to satisfy the order
                            for order in sell_orders:

                                # If action volume to buy is less than order volume to sell
                                if action['volume'] < order['volume']:

                                    # Reduce the action volume from order volume
                                    order['volume'] = order['volume'] - action['volume']

                                    # Append the trade to the trades processed list
                                    self.trades_processed[asset].append(
                                        {'buy_agent': agent, 'volume': action['volume'],
                                         'price': order['limit_min'],
                                         'sell_agent': order['agent'], 'transaction_time': time()})

                                    # Update the last traded price
                                    self.last_traded_price[asset] = order['limit_min']

                                    # Update agent history and holdings of seller
                                    self.agent_history[order['agent']][asset].append(
                                        {'transaction_type': 'sell', 'volume': action['volume'],
                                         'price': order['limit_min'], 'transaction_time': time()})
                                    self.agent_holdings[order['agent']][asset] -= action['volume']

                                    # Update agent history and holdings of buyer
                                    self.agent_history[agent][asset].append(
                                        {'transaction_type': 'buy', 'volume': action['volume'],
                                         'price': order['limit_min'], 'transaction_time': time()})
                                    self.agent_holdings[agent][asset] += action['volume']

                                    # Make action volume to 0
                                    action['volume'] = 0
                                    break

                                # If action volume to buy is greater than order volume to sell
                                elif action['volume'] > order['volume']:

                                    # Reduce the order volume from action volume
                                    action['volume'] = action['volume'] - order['volume']

                                    # Append the trade to the trades processed list
                                    self.trades_processed[asset].append(
                                        {'buy_agent': agent, 'volume': order['volume'],
                                         'price': order['limit_min'],
                                         'sell_agent': order['agent'], 'transaction_time': time()})

                                    # Update the last traded price
                                    self.last_traded_price[asset] = order['limit_min']

                                    # Update agent history and holdings of seller
                                    self.agent_history[order['agent']][asset].append(
                                        {'transaction_type': 'sell', 'volume': order['volume'],
                                         'price': order['limit_min'], 'transaction_time': time()})
                                    self.agent_holdings[order['agent']][asset] -= order['volume']

                                    # Update agent history and holdings of buyer
                                    self.agent_history[agent][asset].append(
                                        {'transaction_type': 'buy', 'volume': order['volume'],
                                         'price': order['limit_min'], 'transaction_time': time()})
                                    self.agent_holdings[agent][asset] += order['volume']

                                    # Remove the order from the sell orders list
                                    self.sell_orders[asset].remove(order)

                                # If action volume to buy is equal to order volume to sell
                                elif action['volume'] == order['volume']:

                                    # Append the trade to the trades processed list
                                    self.trades_processed[asset].append(
                                        {'buy_agent': agent, 'volume': order['volume'],
                                         'price': order['limit_min'],
                                         'sell_agent': order['agent'], 'transaction_time': time()})

                                    # Update the last traded price
                                    self.last_traded_price[asset] = order['limit_min']

                                    # Update agent history and holdings of seller
                                    self.agent_history[order['agent']][asset].append(
                                        {'transaction_type': 'sell', 'volume': order['volume'],
                                         'price': order['limit_min'], 'transaction_time': time()})
                                    self.agent_holdings[order['agent']][asset] -= order['volume']

                                    # Update agent history and holdings of buyer
                                    self.agent_history[agent][asset].append(
                                        {'transaction_type': 'buy', 'volume': order['volume'],
                                         'price': order['limit_min'], 'transaction_time': time()})
                                    self.agent_holdings[agent][asset] += order['volume']

                                    # Remove the order from the sell orders list
                                    self.sell_orders[asset].remove(order)

                                    # Make action volume to 0
                                    action['volume'] = 0
                                    break

                            # If limit buy orders get exhausted append to market buy orders list
                            if action['volume'] > 0:
                                self.market_buy_orders[asset].append(
                                    {'agent': agent, 'order_time': time(), 'volume': action['volume']})

                        # If no limit sell orders are present append to market buy orders list
                        else:
                            self.market_buy_orders[asset].append(
                                {'agent': agent, 'order_time': time(), 'volume': action['volume']})

                # Limit order
                elif action['order_type'] == 'limit':

                    # Hold action
                    if action['action'] == 0:
                        pass

                    # Sell action
                    elif action['action'] == 1:

                        # Check for pending market buy orders
                        if self.market_buy_orders[asset]:
                            market_buy_orders = sorted(self.market_buy_orders[asset],
                                                       key=itemgetter('order_time'))
                            for order in market_buy_orders:

                                # If action volume to sell is less than market order volume to buy
                                if action['volume'] < order['volume']:

                                    # Reduce the action volume from order volume
                                    order['volume'] = order['volume'] - action['volume']

                                    # Append the trade to the trades processed list
                                    self.trades_processed[asset].append(
                                        {'buy_agent': order['agent'], 'volume': action['volume'],
                                         'price': action['limit_min'],
                                         'sell_agent': agent, 'transaction_time': time()})

                                    # Update the last traded price
                                    self.last_traded_price[asset] = action['limit_min']

                                    # Update agent history and holdings of buyer
                                    self.agent_history[order['agent']][asset].append(
                                        {'transaction_type': 'buy', 'volume': action['volume'],
                                         'price': action['limit_min'], 'transaction_time': time()})
                                    self.agent_holdings[order['agent']][asset] += action['volume']

                                    # Update agent history and holdings of seller
                                    self.agent_history[agent][asset].append(
                                        {'transaction_type': 'sell', 'volume': action['volume'],
                                         'price': action['limit_min'], 'transaction_time': time()})
                                    self.agent_holdings[agent][asset] -= action['volume']

                                    # Make action volume to 0
                                    action['volume'] = 0

                                    break

                                # If action volume to sell is greater than market order volume to buy
                                elif action['volume'] > order['volume']:

                                    # Reduce the order volume from action volume
                                    action['volume'] = action['volume'] - order['volume']

                                    # Append the trade to the trades processed list
                                    self.trades_processed[asset].append(
                                        {'buy_agent': order['agent'], 'volume': order['volume'],
                                         'price': action['limit_min'],
                                         'sell_agent': agent, 'transaction_time': time()})

                                    # Update agent history of buyer
                                    self.agent_history[order['agent']][asset].append(
                                        {'transaction_type': 'buy', 'volume': order['volume'],
                                         'price': action['limit_min'], 'transaction_time': time()})
                                    self.agent_holdings[order['agent']][asset] += order['volume']

                                    # Update agent history of seller
                                    self.agent_history[agent][asset].append(
                                        {'transaction_type': 'sell', 'volume': order['volume'],
                                         'price': action['limit_min'], 'transaction_time': time()})
                                    self.agent_holdings[agent][asset] -= order['volume']

                                    # Update the last traded price
                                    self.last_traded_price[asset] = action['limit_min']

                                    # Remove the order from the market buy orders list
                                    self.market_buy_orders[asset].remove(order)

                                # If action volume to sell is equal to market order volume to buy
                                elif action['volume'] == order['volume']:

                                    # Append the trade to the trades processed list
                                    self.trades_processed[asset].append(
                                        {'buy_agent': order['agent'], 'volume': order['volume'],
                                         'price': action['limit_min'],
                                         'sell_agent': agent, 'transaction_time': time()})

                                    # Update the last traded price
                                    self.last_traded_price[asset] = action['limit_min']

                                    # Update agent history and holdings of buyer
                                    self.agent_history[order['agent']][asset].append(
                                        {'transaction_type': 'buy', 'volume': order['volume'],
                                         'price': action['limit_min'], 'transaction_time': time()})
                                    self.agent_holdings[order['agent']][asset] += order['volume']

                                    # Update agent history and holdings of seller
                                    self.agent_history[agent][asset].append(
                                        {'transaction_type': 'sell', 'volume': order['volume'],
                                         'price': action['limit_min'], 'transaction_time': time()})
                                    self.agent_holdings[agent][asset] -= order['volume']

                                    # Remove the order from the market buy orders list
                                    self.market_buy_orders[asset].remove(order)

                                    # Make action volume to 0
                                    action['volume'] = 0
                                    break

                        # If there are no pending market buy orders
                        if not self.market_buy_orders[asset]:

                            # Check if limit buy orders are present
                            if self.buy_orders[asset]:
                                buy_orders_by_price = sorted(self.buy_orders[asset],
                                                             key=itemgetter('limit_max'), reverse=True)

                                # If price of the action is greater the max priced order present
                                if action['limit_min'] > buy_orders_by_price[0]['limit_max']:
                                    self.sell_orders[asset].append(
                                        {'agent': agent, 'volume': action['volume'],
                                         'limit_min': action['limit_min'], 'order_time': time()})

                                # Else if the price of the action is lesser than the max priced order present
                                else:

                                    # Sort buy orders by time
                                    buy_orders = sorted(self.buy_orders[asset],
                                                        key=itemgetter('order_time'))

                                    for order in buy_orders:

                                        # If action limit amount is lesser than the order limit amount
                                        if action['limit_min'] <= order['limit_max']:

                                            # If action limit volume to sell is less than order limit volume to buy
                                            if action['volume'] < order['volume']:

                                                # Reduce the action volume from order volume
                                                order['volume'] = order['volume'] - action['volume']

                                                # Append the trade to the trades processed list
                                                self.trades_processed[asset].append(
                                                    {'buy_agent': order['agent'], 'volume': action['volume'],
                                                     'price': order['limit_max'],
                                                     'sell_agent': agent, 'transaction_time': time()})

                                                # Update the last traded price
                                                self.last_traded_price[asset] = order['limit_max']

                                                # Update agent history and holdings of buyer
                                                self.agent_history[order['agent']][asset].append(
                                                    {'transaction_type': 'buy', 'volume': action['volume'],
                                                     'price': order['limit_max'], 'transaction_time': time()})
                                                self.agent_holdings[order['agent']][asset] += action['volume']

                                                # Update agent history and holdings of seller
                                                self.agent_history[agent][asset].append(
                                                    {'transaction_type': 'sell', 'volume': action['volume'],
                                                     'price': order['limit_max'], 'transaction_time': time()})
                                                self.agent_holdings[agent][asset] -= action['volume']

                                                # Make action volume to 0
                                                action['volume'] = 0
                                                break

                                            # If action limit volume to sell is greater than order limit volume to buy
                                            elif action['volume'] > order['volume']:

                                                # Reduce the order volume from action volume
                                                action['volume'] = action['volume'] - order['volume']

                                                # Append the trade to the trades processed list
                                                self.trades_processed[asset].append(
                                                    {'buy_agent': order['agent'], 'volume': order['volume'],
                                                     'price': order['limit_max'],
                                                     'sell_agent': agent, 'transaction_time': time()})

                                                # Update the last traded price
                                                self.last_traded_price[asset] = order['limit_max']

                                                # Update agent history and holdings of buyer
                                                self.agent_history[order['agent']][asset].append(
                                                    {'transaction_type': 'buy', 'volume': order['volume'],
                                                     'price': order['limit_max'], 'transaction_time': time()})
                                                self.agent_holdings[order['agent']][asset] += order['volume']

                                                # Update agent history and holdings of seller
                                                self.agent_history[agent][asset].append(
                                                    {'transaction_type': 'sell', 'volume': order['volume'],
                                                     'price': order['limit_max'], 'transaction_time': time()})
                                                self.agent_holdings[agent][asset] -= order['volume']

                                                # Remove the order from the buy orders list
                                                self.buy_orders[asset].remove(order)

                                            # If action limit volume to sell is equal to order limit volume to buy
                                            elif action['volume'] == order['volume']:

                                                # Append the trade to the trades processed list
                                                self.trades_processed[asset].append(
                                                    {'buy_agent': order['agent'], 'volume': order['volume'],
                                                     'price': order['limit_max'],
                                                     'sell_agent': agent, 'transaction_time': time()})

                                                # Update the last traded price
                                                self.last_traded_price[asset] = order['limit_max']

                                                # Update agent history and holdings of buyer
                                                self.agent_history[order['agent']][asset].append(
                                                    {'transaction_type': 'buy', 'volume': order['volume'],
                                                     'price': order['limit_max'], 'transaction_time': time()})
                                                self.agent_holdings[order['agent']][asset] += order['volume']

                                                # Update agent history and holdings of seller
                                                self.agent_history[agent][asset].append(
                                                    {'transaction_type': 'sell', 'volume': order['volume'],
                                                     'price': order['limit_max'], 'transaction_time': time()})
                                                self.agent_holdings[agent][asset] -= order['volume']

                                                # Remove the order from the buy orders list
                                                self.buy_orders[asset].remove(order)

                                                # Make action volume to 0
                                                action['volume'] = 0
                                                break

                                    # If limit buy order cannot be satisfied for remaining volume
                                    if action['volume'] > 0:
                                        self.sell_orders[asset].append(
                                            {'agent': agent, 'volume': action['volume'],
                                             'limit_min': action['limit_min'], 'order_time': time()})

                            # If no limit buy orders are present append to limit sell orders list
                            else:
                                self.sell_orders[asset].append(
                                    {'agent': agent, 'volume': action['volume'],
                                     'limit_min': action['limit_min'], 'order_time': time()})

                    # Buy action
                    elif action['action'] == 2:

                        # Check for pending market sell orders
                        if self.market_sell_orders[asset]:
                            market_sell_orders = sorted(self.market_sell_orders[asset],
                                                        key=itemgetter('order_time'))
                            for order in market_sell_orders:

                                # If action volume to buy is less than market order volume to sell
                                if action['volume'] < order['volume']:

                                    # Reduce the action volume from order volume
                                    order['volume'] = order['volume'] - action['volume']

                                    # Append the trade to the trades processed list
                                    self.trades_processed[asset].append(
                                        {'buy_agent': agent, 'volume': action['volume'],
                                         'price': action['limit_max'],
                                         'sell_agent': order['agent'], 'transaction_time': time()})

                                    # Update agent history and holdings of seller
                                    self.agent_history[order['agent']][asset].append(
                                        {'transaction_type': 'sell', 'volume': action['volume'],
                                         'price': action['limit_max'], 'transaction_time': time()})
                                    self.agent_holdings[order['agent']][asset] -= action['volume']

                                    # Update agent history and holdings of buyer
                                    self.agent_history[agent][asset].append(
                                        {'transaction_type': 'buy', 'volume': action['volume'],
                                         'price': action['limit_max'], 'transaction_time': time()})
                                    self.agent_holdings[agent][asset] += action['volume']

                                    # Update the last traded price
                                    self.last_traded_price[asset] = action['limit_max']

                                    # Make action volume to 0
                                    action['volume'] = 0
                                    break

                                # If action volume to buy is greater than market order volume to sell
                                elif action['volume'] > order['volume']:

                                    # Reduce the order volume from action volume
                                    action['volume'] = action['volume'] - order['volume']

                                    # Append the trade to the trades processed list
                                    self.trades_processed[asset].append(
                                        {'buy_agent': agent, 'volume': order['volume'],
                                         'price': action['limit_max'],
                                         'sell_agent': order['agent'], 'transaction_time': time()})

                                    # Update agent history and holdings of seller
                                    self.agent_history[order['agent']][asset].append(
                                        {'transaction_type': 'sell', 'volume': order['volume'],
                                         'price': action['limit_max'], 'transaction_time': time()})
                                    self.agent_holdings[order['agent']][asset] -= order['volume']

                                    # Update agent history and holdings of buyer
                                    self.agent_history[agent][asset].append(
                                        {'transaction_type': 'buy', 'volume': order['volume'],
                                         'price': action['limit_max'], 'transaction_time': time()})
                                    self.agent_holdings[agent][asset] += order['volume']

                                    # Update the last traded price
                                    self.last_traded_price[asset] = action['limit_max']

                                    # Remove the order from the market sell orders list
                                    self.market_sell_orders[asset].remove(order)

                                # If action volume to buy is equal to market order volume to sell
                                elif action['volume'] == order['volume']:

                                    # Append the trade to the trades processed list
                                    self.trades_processed[asset].append(
                                        {'buy_agent': agent, 'volume': order['volume'],
                                         'price': action['limit_max'],
                                         'sell_agent': order['agent'], 'transaction_time': time()})

                                    # Update agent history and holdings of seller
                                    self.agent_history[order['agent']][asset].append(
                                        {'transaction_type': 'sell', 'volume': order['volume'],
                                         'price': action['limit_max'], 'transaction_time': time()})
                                    self.agent_holdings[order['agent']][asset] -= order['volume']

                                    # Update agent history and holdings of buyer
                                    self.agent_history[agent][asset].append(
                                        {'transaction_type': 'buy', 'volume': order['volume'],
                                         'price': action['limit_max'], 'transaction_time': time()})
                                    self.agent_holdings[agent][asset] += order['volume']

                                    # Update the last traded price
                                    self.last_traded_price[asset] = action['limit_max']

                                    # Remove the order from the market sell orders list
                                    self.market_sell_orders[asset].remove(order)

                                    # Make action volume to 0
                                    action['volume'] = 0
                                    break

                        # If there are no pending market sell orders
                        if not self.market_sell_orders[asset]:

                            # Check if limit sell orders are present
                            if self.sell_orders[asset]:
                                sell_orders_by_price = sorted(self.sell_orders[asset],
                                                              key=itemgetter('limit_min'))

                                # If price of the action is greater the max priced order present
                                if action['limit_max'] < sell_orders_by_price[0]['limit_min']:
                                    self.buy_orders[asset].append(
                                        {'agent': agent, 'volume': action['volume'],
                                         'limit_max': action['limit_max'], 'order_time': time()})

                                # Else if the price of the action is lesser than the max priced order present
                                else:

                                    # Sort buy orders by time
                                    sell_orders = sorted(self.sell_orders[asset],
                                                         key=itemgetter('order_time'))

                                    for order in sell_orders:

                                        # If action limit amount is lesser than the order limit amount
                                        if action['limit_max'] >= order['limit_min']:

                                            # If action limit volume to buy is less than order limit volume to sell
                                            if action['volume'] < order['volume']:

                                                # Reduce the action volume from order volume
                                                order['volume'] = order['volume'] - action['volume']

                                                # Append the trade to the trades processed list
                                                self.trades_processed[asset].append(
                                                    {'buy_agent': agent, 'volume': action['volume'],
                                                     'price': order['limit_min'],
                                                     'sell_agent': order['agent'], 'transaction_time': time()})

                                                # Update the last traded price
                                                self.last_traded_price[asset] = order['limit_min']

                                                # Update agent history and holdings of seller
                                                self.agent_history[order['agent']][asset].append(
                                                    {'transaction_type': 'sell', 'volume': action['volume'],
                                                     'price': order['limit_min'], 'transaction_time': time()})
                                                self.agent_holdings[order['agent']][asset] -= action['volume']

                                                # Update agent history and holdings of buyer
                                                self.agent_history[agent][asset].append(
                                                    {'transaction_type': 'buy', 'volume': action['volume'],
                                                     'price': order['limit_min'], 'transaction_time': time()})
                                                self.agent_holdings[agent][asset] += action['volume']

                                                # Make action volume to 0
                                                action['volume'] = 0
                                                break

                                            # If action limit volume to buy is greater than order limit volume to sell
                                            elif action['volume'] > order['volume']:

                                                # Reduce the order volume from action volume
                                                action['volume'] = action['volume'] - order['volume']

                                                # Append the trade to the trades processed list
                                                self.trades_processed[asset].append(
                                                    {'buy_agent': agent, 'volume': order['volume'],
                                                     'price': order['limit_min'],
                                                     'sell_agent': order['agent'], 'transaction_time': time()})

                                                # Update the last traded price
                                                self.last_traded_price[asset] = order['limit_min']

                                                # Update agent history and holdings of seller
                                                self.agent_history[order['agent']][asset].append(
                                                    {'transaction_type': 'sell', 'volume': order['volume'],
                                                     'price': order['limit_min'], 'transaction_time': time()})
                                                self.agent_holdings[order['agent']][asset] -= order['volume']

                                                # Update agent history and holdings of buyer
                                                self.agent_history[agent][asset].append(
                                                    {'transaction_type': 'buy', 'volume': order['volume'],
                                                     'price': order['limit_min'], 'transaction_time': time()})
                                                self.agent_holdings[agent][asset] += order['volume']

                                                # Remove the order from the sell orders list
                                                self.sell_orders[asset].remove(order)

                                            # If action limit volume to buy is equal to order limit volume to sell
                                            elif action['volume'] == order['volume']:

                                                # Append the trade to the trades processed list
                                                self.trades_processed[asset].append(
                                                    {'buy_agent': agent, 'volume': order['volume'],
                                                     'price': order['limit_min'],
                                                     'sell_agent': order['agent'], 'transaction_time': time()})

                                                # Update the last traded price
                                                self.last_traded_price[asset] = order['limit_min']

                                                # Update agent history and holdings of seller
                                                self.agent_history[order['agent']][asset].append(
                                                    {'transaction_type': 'sell', 'volume': order['volume'],
                                                     'price': order['limit_min'], 'transaction_time': time()})
                                                self.agent_holdings[order['agent']][asset] -= order['volume']

                                                # Update agent history and holdings of buyer
                                                self.agent_history[agent][asset].append(
                                                    {'transaction_type': 'buy', 'volume': order['volume'],
                                                     'price': order['limit_min'], 'transaction_time': time()})
                                                self.agent_holdings[agent][asset] += order['volume']

                                                # Remove the order from the sell orders list
                                                self.sell_orders[asset].remove(order)

                                                # Make action volume to 0
                                                action['volume'] = 0
                                                break

                                    # If limit buy order cannot be satisfied for remaining volume
                                    if action['volume'] > 0:
                                        self.buy_orders[asset].append(
                                            {'agent': agent, 'volume': action['volume'],
                                             'limit_max': action['limit_max'], 'order_time': time()})

                            # If no limit buy orders are present append to limit buy orders list
                            else:
                                self.buy_orders[asset].append(
                                    {'agent': agent, 'volume': action['volume'],
                                     'limit_max': action['limit_max'], 'order_time': time()})

        orders_bids = sorted(self.buy_orders["asset_0"], key=lambda k: k['limit_max'])
        orders_asks = sorted(self.sell_orders["asset_0"], key=lambda k: k['limit_min'])
        rewards = self.infer("asset_0", "reward")
        holdings = self.agent_holdings
        # self.red_db.publish('data', json.dumps(
        #     {"ltp": float(self.last_traded_price['asset_0']), "bids": str(orders_bids[-50:]),
        #      "asks": str(orders_asks[:50]), "trades": self.trades_processed["asset_0"][-10:],
        #      "rewards": [] if rewards is None else rewards, "holdings": [] if holdings is None else holdings}))
        # return self.rewards

    def infer(self, asset, query):
        reward = 0
        if query == "reward":
            for agent in self.agents:
                if len(self.agent_actions[agent][asset]) > 2:
                    action = self.agent_actions[agent][asset][-2]["action"]
                else:
                    return None

                if self.agent_history[agent][asset]:
                    prev_ltp = self.trades_processed[asset][-2]['price']
                    returns = (prev_ltp - self.last_traded_price[asset]) / prev_ltp
                    returns = returns - (self.transaction_cost * returns)
                else:
                    # prev_ltp = 0.0
                    returns = 0

                if (returns > 0 and action == 2) or (
                        returns < 0 and action == 1) or (
                        returns > 0 and action == 0):
                    reward = 1

                elif (returns < 0 and action == 2) or (
                        returns > 0 and action == 1) or (returns < 0 and action == 0):
                    reward = -1

                elif (returns == 0 and action == 0) or (
                        returns == 0 and action == 1) or (returns == 0 and action == 2):
                    reward = 0

                elif np.isnan(returns):
                    reward = 0
                try:
                    self.rewards[agent][asset] = reward
                except:
                    print(action, type(returns))
            return self.rewards
        elif query == "history":
            for agent in self.agents:
                return self.agent_history[agent][asset]
        elif query == "bs_log":
            return self.buy_orders, self.sell_orders

    def reset(self):
        ...

    def render(self, mode='human', close=False):
        ...
