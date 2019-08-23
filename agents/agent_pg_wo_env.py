# Policy gradient implementation for stock market agent
import datetime

import tensorflow as tf
import numpy as np
# from rl_gym import envs
import pandas as pd
import sys
import os
from tensorflow.python import debug as tf_debug
from collections import OrderedDict
import shutil

tf.reset_default_graph()
agents = ["agent1", "agent2"]
# env = envs.fin_env_pol_gra.FinEnvPolicyGradientTest(agents)
# assets = env.assets
action_size = 3
learning_rate = 1e-4
data = pd.read_csv('C:\Users\myhom\Projects\Python codes\Strategy-test-env\Nifty_price_new.csv')
data['Ret'] = data['Close'].pct_change()
data['Symbol'] = 's1'
data = data.dropna()
# print(data.shape)
# sys.exit()
# data = data.iloc[1:]
batch_size = 10
lookback = 100
rollahead = 1
max_iterations = data.shape[0] - 102
gamma = 0.99
no_of_feat = 5
ticks_to_consider = 750

if os.path.exists('../tensorboard/pg/batch'):
    shutil.rmtree('../tensorboard/pg/batch')


def rolling_backtest_index(nsamples, look_back, roll_ahead):
    coll = []
    for i in range(look_back + roll_ahead, nsamples + roll_ahead, roll_ahead):
        col = [i - look_back - roll_ahead, i - roll_ahead, min(i, nsamples)]
        coll.append(col)
    return coll


def num_of_trades(signals_i):
    sigs = np.sign(signals_i)
    in_trd = False
    prev_trade = 0
    cnt_trd = 0
    for i in range(len(sigs)):
        if in_trd:
            if prev_trade == 1:
                if sigs[i] == 0:
                    in_trd = False
                elif sigs[i] == -1:
                    cnt_trd += 1
            elif prev_trade == -1:
                if sigs[i] == 0:
                    in_trd = False
                elif sigs[i] == 1:
                    cnt_trd += 1
        else:
            if sigs[i] == 1:
                cnt_trd += 1
                in_trd = True
            elif sigs[i] == -1:
                cnt_trd += 1
                in_trd = True
        prev_trade = sigs[i]
    return cnt_trd


def discount_and_normalize_rewards(episode_rewards_list):
    discounted_iteration_rewards = np.zeros_like(episode_rewards_list)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards_list))):
        cumulative = cumulative * gamma + episode_rewards_list[i]
        discounted_iteration_rewards[i] = cumulative

    mean = np.mean(discounted_iteration_rewards)
    std = np.std(discounted_iteration_rewards)
    discounted_iteration_rewards = (discounted_iteration_rewards - mean) / std
    return discounted_iteration_rewards


def my_filter_callable(datum, tensor):
    # A filter that detects zero-valued scalars.
    return len(tensor.shape) == 0 and tensor == 0.0


with tf.name_scope('policy_gradient'):
    price_vol_input = tf.placeholder(dtype=tf.float16, shape=(None, lookback, no_of_feat), name="price_vol_input")
    actions = tf.placeholder(dtype=tf.float16, shape=(None, action_size), name="actions")
    reward_p = tf.placeholder(tf.float16, shape=(None,), name="reward")

    with tf.name_scope("fc1"):
        fc1 = tf.contrib.layers.fully_connected(inputs=price_vol_input, num_outputs=75, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                normalizer_fn=tf.contrib.layers.batch_norm)

    with tf.name_scope("fc2"):
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=50, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("fc3"):
        fc3 = tf.contrib.layers.fully_connected(inputs=fc2, num_outputs=25, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                normalizer_fn=tf.contrib.layers.batch_norm)

    with tf.name_scope("fc4"):
        fc4 = tf.contrib.layers.fully_connected(inputs=fc3, num_outputs=12, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("fc5"):
        fc5 = tf.contrib.layers.fully_connected(inputs=fc4, num_outputs=action_size, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                normalizer_fn=tf.contrib.layers.batch_norm)

    with tf.name_scope("fc6"):
        fc6 = tf.contrib.layers.fully_connected(inputs=tf.reshape(fc5, [-1, 1, action_size * 100]),
                                                num_outputs=action_size,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc6)

    with tf.name_scope("loss"):
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc6, labels=actions)
        loss = tf.reduce_mean(neg_log_prob * reward_p)
        # loss = tf.reduce_mean((reward_p * fc4) - actions)

    with tf.name_scope("train"):
        train_opt = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

    # check = tf.add_check_numerics_ops()

writer = tf.summary.FileWriter("../tensorboard/pg/batch")
tf.summary.scalar("Loss", loss)
tf.summary.scalar("Reward", tf.reduce_mean(reward_p))
write_op = tf.summary.merge_all()

episode_states, episode_actions, episode_rewards = [], [], []
window_index = 0

saver = tf.train.Saver(save_relative_paths=True)
windows = rolling_backtest_index(data.shape[0], lookback, rollahead)
action_considered = 0
predicted_values = []
prev_actions = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter('my_filter', my_filter_callable)

    writer.add_graph(tf.get_default_graph())

    for iteration in range(max_iterations):
        train_data = data.iloc[windows[window_index][0]:windows[window_index][1], :]
        next_tick_ret = np.sign(data.iloc[windows[window_index][1] + 1, :]['Ret'])
        next_tick_date = data.iloc[windows[window_index][1] + 1, :]['Date/Time']
        next_tick_symbol = data.iloc[windows[window_index][1] + 1, :]['Symbol']
        next_tick_price = data.iloc[windows[window_index][1] + 1, :]['Close']
        # print(data.shape)
        train_data = train_data[
            ['Highest Close of five bar < Close', 'Range', 'Low < Lowest(low)', 'Average > Close', 'Ret']]
        # train_data.drop(['Date/Time', 'Symbol', 'Close'], axis=1, inplace=True)
        train_data = (train_data - train_data.mean()) / train_data.std()
        state = train_data
        reward = 0.0
        action_prob_dist = sess.run(action_distribution, feed_dict={price_vol_input: [np.array(state)]})
        value = np.argmax(action_prob_dist.ravel())
        # print(np.shape(action_prob_dist))

        # value = np.random.choice(range(action_prob_dist.shape[2]), p=action_prob_dist.ravel())
        action_ = np.zeros(action_size)
        action_[value] = 1
        # print(action_)
        # print(action_prob_dist)
        # print(np.random.choice(range(action_prob_dist.shape[2]), p=action_prob_dist.ravel()))

        if value == 0:
            # hodl
            action_considered = 0
        elif value == 1:
            # buy
            action_considered = 1
        elif value == 2:
            # sell
            action_considered = -1
        action = action_
        if len(prev_actions) > ticks_to_consider:
            prev_actions.pop(0)

        prev_actions.append(action_considered)

        num_trd_last = num_of_trades(prev_actions)

        if action_considered == 0:
            if next_tick_ret == -1:
                reward = 1.0
            elif next_tick_ret == 1:
                reward = -1.0
        else:
            if action_considered == next_tick_ret:
                reward = 5.0
            else:
                reward = -5.0
        # print(action, value)
        reward = reward - num_trd_last
        # if num_trd_last > 10:
        #     reward = [[reward[0][0] - num_trd_last * 10]]
        # else:
        #     reward = [[reward[0][0] + 4.0]]
        episode_rewards.append(reward)
        episode_actions.append(action)
        episode_states.append(np.array(state))
        window_index += 1

        print("===========================")
        print("Iteration ", iteration)
        print("Num of trades ", num_trd_last)
        print("Action considered ", action_considered)
        print("Reward ", reward)
        print(action_prob_dist.ravel())
        if iteration > 2000:
            predicted_values.append([next_tick_symbol, next_tick_date, next_tick_price, action_considered])

        if iteration % 20000 == 0:
            pr_df = pd.DataFrame(predicted_values, columns=['Symbol', 'Date', 'Mid', 'Signal'])
            # pr_df.index = map(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y %H:%M"), pr_df.index)
            # del pr_df['Date']
            pr_df.to_excel('signals.xlsx')

        if len(episode_rewards) == batch_size:
            # episode_rewards = discount_and_normalize_rewards(episode_rewards)
            # print(episode_actions)
            neg_log_prob_, loss_, summary, _ = sess.run([neg_log_prob, loss, write_op, train_opt],
                                                        feed_dict={
                                                            price_vol_input: np.stack(np.array(episode_states)),
                                                            reward_p: np.stack(np.array(episode_rewards)),
                                                            actions: np.stack(np.array(episode_actions))
                                                        })

            overall_rew = np.mean(np.ravel(episode_rewards))
            print("==========================================")
            print("Iteration: ", iteration)
            print("Reward: ", overall_rew)
            print("Loss: ", loss_)
            if loss_ == np.float16("-inf") or np.isnan(loss_):
                sys.exit()
            # print(check_)
            episode_states, episode_actions, episode_rewards = [], [], []

            writer.add_summary(summary, iteration)
            writer.flush()

        if iteration % 25 == 0:
            saver.save(sess, "./models/batch/model.ckpt")
            print("Model saved")

    pr_df = pd.DataFrame(predicted_values, columns=['Symbol', 'Date', 'Mid', 'Signal'])
    # pr_df.index = map(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y %H:%M"), pr_df.index)
    # del pr_df['Date']
    pr_df.to_excel('signals.xlsx')
