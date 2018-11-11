import numpy as np
import random
import os
import math
import copy
import datetime
from collections import Counter, deque
from statistics import median, mean
import tensorflow as tf
import gym
import gym.spaces
from gym import wrappers

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "CartPole-v0", "Gym enviroment name.")
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes.')
flags.DEFINE_boolean("render", False, "Enable game rendering.")
flags.DEFINE_boolean("monitor", False, "Enable game monitoring.")

class PolicyGradient:
    def __init__(self, session, obs_size, action_size):
        self.time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session = session
        self.obs_size = obs_size
        self.action_size = action_size
        self.learning_rate = 0.02
        self.gamma = 0.99
        self.tick = 0

        # storage for transitions
        self.ep_obs, self.ep_actions, self.ep_rewards = [], [], []

        # create model
        self.create_model()

        # create training
        self.create_trainig()

        # init tf
        self.session.run(tf.global_variables_initializer())

        # saver
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
           self.saver.restore(self.session, checkpoint.model_checkpoint_path)
           print(f'Successfully loaded: {checkpoint.model_checkpoint_path}')
        else:
           print('Could not find old network weights')

    def train(self):
        self.session.run(self.train_op, feed_dict={
            self.obs_ph: np.vstack(self.ep_obs),
            self.action_ph: np.array(self.ep_actions),
            self.discounted_rewards_ph: self.discounted_rewards,
        })

    def calc_discounted_rewards(self, rewards, gamma):
        discounted_rewards = np.zeros_like(rewards)
        current_reward = 0
        for i in reversed(range(0, len(rewards))):
            current_reward = current_reward * gamma + rewards[i]
            discounted_rewards[i] = current_reward

        # normalize discounted rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        return discounted_rewards

    def process(self, obs, action, reward, done):
        # save transitions
        self.ep_obs.append(obs)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)

        if done:
            # calc discounted rewards
            self.discounted_rewards = self.calc_discounted_rewards(self.ep_rewards, self.gamma)

            # train
            self.train()

            # reset transistion storage
            self.ep_obs, self.ep_actions, self.ep_rewards = [], [], []

        # save network every 1000 iteration
        if self.tick > 0 and self.tick % 1000 == 0:
            self.saver.save(self.session, f'saved_networks/dqn-{self.time}', global_step = self.tick)

        self.tick += 1


    def get_action(self, obs, training=True):
        action_prob = self.session.run(self.action_prob, feed_dict={self.obs_ph: obs[np.newaxis, :]})
        action = np.random.choice(range(action_prob.shape[1]), p=action_prob.ravel())

        return action

    def create_trainig(self):
        with tf.name_scope('loss'):
            neg_log_prob = tf.reduce_sum(-tf.log(self.action_prob) * tf.one_hot(self.action_ph, self.action_size), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_rewards_ph)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def create_model(self):
        with tf.name_scope('inputs'):
            self.obs_ph = tf.placeholder(tf.float32, [None, self.obs_size])
            self.action_ph = tf.placeholder(tf.int32, [None])
            self.discounted_rewards_ph = tf.placeholder(tf.float32, [None])

        hidden1 = tf.layers.dense(
            inputs=self.obs_ph,
            units=24,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1)
        )

        hidden2 = tf.layers.dense(
            inputs=hidden1,
            units=24,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1)
        )

        self.action_prob = tf.layers.dense(
            inputs=hidden2,
            units=self.action_size,
            activation=tf.nn.softmax,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1)
        )

def main(argv=None):
    # create enviroment
    env = gym.make(FLAGS.env_name)

    if FLAGS.monitor:
        env = wrappers.Monitor(env, f'video/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    # use cpu only
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

    with tf.Session(config=config) as sess:
        pgAgent = PolicyGradient(sess, env.observation_space.shape[0], env.action_space.n)

        for i in range(FLAGS.num_episodes):
            done = False
            obs = env.reset()
            total_reward = 0

            while not done:
                if FLAGS.render:
                    env.render()

                # get action
                action = pgAgent.get_action(obs)

                # take action
                new_obs, reward, done, info = env.step(action)
                total_reward += reward

                # process transition
                pgAgent.process(obs, action, reward, done)

                # episode done?
                if done:
                    print(f'Episode: {i}, Reward: {total_reward}')

                obs = new_obs

if __name__ == "__main__":
    tf.app.run()
