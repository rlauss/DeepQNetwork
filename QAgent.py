import numpy as np
import random
import os
import math
import copy
import datetime
from collections import Counter, deque
from statistics import median, mean
import tensorflow as tf
from replay_memory import ReplayMemory
import matplotlib.pyplot as plt
import cv2

OBSERVE = 50000
EXPLORE = 1000000
TRAIN_FREQ = 4
UPDATE_TIME = 10000
FINAL_EPSILON = 0.1
INITIAL_EPSILON = 0.1#1.0
LOGDIR = 'logs/'

class QAgent:
    def __init__(self, session):
        self.time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session = session
        self.action_size = 4
        self.gamma = 0.99
        self.epsilon = INITIAL_EPSILON
        self.batch_size = 32
        self.learning_rate = 0.00001

        self.replay_mem = ReplayMemory(size=1000000, 
                                       frame_height=84, 
                                       frame_width=84,
                                       agent_history_length=4, 
                                       batch_size=32)
        self.tick = 0
        self.episode = 0
        self.total_reward = 0
        self.last_n_game_reward = deque(maxlen=100)

        # create q network
        self.state_input, self.q_values, self.best_action = self.create_model('main')
        
        # create target q network
        self.state_input_t, self.q_values_t, self.best_action_t = self.create_model('target')

        # update dqn vars
        self.main_dqn_vars = tf.trainable_variables(scope='main')
        self.target_dqn_vars = tf.trainable_variables(scope='target')

        self.update_ops = []
        for i, var in enumerate(self.main_dqn_vars):
            copy_op = self.target_dqn_vars[i].assign(var.value())
            self.update_ops.append(copy_op)

        # create training
        self.create_trainig()

        # init
        self.session.run(tf.global_variables_initializer())
        #self.update_target_network()

        # saver
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
           self.saver.restore(self.session, checkpoint.model_checkpoint_path)
           print(f'Successfully loaded: {checkpoint.model_checkpoint_path}')
        else:
           print('Could not find old network weights')

    def create_trainig(self):
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.action_size, dtype=tf.float32)), axis=1)
        
        # loss
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.target_q, predictions=self.Q))

        # train
        with tf.variable_scope("training"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)


    def train(self):
        states, actions, rewards, new_states, terminal_flags = self.replay_mem.get_minibatch()

        # test = states[0]
        # print(test.dtype)
        # print(test.shape)
        # fig = plt.figure(figsize=(1, 4))
        # fig.add_subplot(1, 4, 1)
        # plt.imshow(test[:,:,0])
        # fig.add_subplot(1, 4, 2)
        # plt.imshow(test[:,:,1])
        # fig.add_subplot(1, 4, 3)
        # plt.imshow(test[:,:,2])
        # fig.add_subplot(1, 4, 4)
        # plt.imshow(test[:,:,3])
        # plt.show()
        # ttt

        # main dqn: predict best action for new state
        arg_q_max = self.session.run(self.best_action, feed_dict={self.state_input:new_states})

        # target qdn: predict Q(s', a)
        q_vals = self.session.run(self.q_values_t, feed_dict={self.state_input_t:new_states})

        # double q
        double_q = q_vals[range(self.batch_size), arg_q_max]

        # calc bellman equation
        target_q = rewards + (self.gamma * double_q * (1 - terminal_flags))

        # train main dqn
        loss, _ = self.session.run([self.loss, self.train_op], feed_dict={self.state_input:states, 
                                                                     self.target_q:target_q, 
                                                                     self.action:actions})

    def process(self, next_frame, action, reward, done, terminal_life_lost):
        self.replay_mem.add_experience(action=action, 
                                        frame=next_frame[:, :, 0],
                                        reward=reward, 
                                        terminal=terminal_life_lost)
        
        if self.tick > OBSERVE and self.tick % TRAIN_FREQ:
            self.train()

        # update target network
        if self.tick > OBSERVE and self.tick % UPDATE_TIME == 0:
            print('update target network')
            self.update_target_network()

        # save network every 100000 iteration
        if self.tick > 0 and self.tick % 100000 == 0:
            self.saver.save(self.session, f'saved_networks/dqn-{self.time}', global_step = self.tick)

        self.total_reward += reward

        if done:
            self.last_n_game_reward.append(self.total_reward)
            # print
            print(f'Episode: {self.episode}, Reward: {self.total_reward}, Avg. Reward: {np.mean(self.last_n_game_reward)}, Epsilon: {self.epsilon}, Step: {self.tick}')
            self.episode += 1
            self.total_reward = 0

        self.tick += 1


    def get_action(self, state, training=True):
        if training:
            if random.random() < self.epsilon:
                action = np.random.randint(0, 4)
            else:
                action = self.session.run(self.best_action, feed_dict={self.state_input:[state]})[0]

            if self.epsilon > FINAL_EPSILON and self.tick > OBSERVE:
                self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE
        else:
            action = self.session.run(self.best_action, feed_dict={self.state_input:[state]})[0]   

        return action

    def update_target_network(self):
        for copy_op in self.update_ops:
            self.session.run(copy_op)

    def create_model(self, name = 'main'):
        with tf.variable_scope(name):
            stateInput = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32)
        
            # Normalizing the input
            inputscaled = stateInput / 255
        
            # Convolutional layers
            conv1 = tf.layers.conv2d(
                inputs=inputscaled, filters=32, kernel_size=[8, 8], strides=4,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False)
            conv2 = tf.layers.conv2d(
                inputs=conv1, filters=64, kernel_size=[4, 4], strides=2, 
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False)
            conv3 = tf.layers.conv2d(
                inputs=conv2, filters=64, kernel_size=[3, 3], strides=1, 
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False)
            conv4 = tf.layers.conv2d(
                inputs=conv3, filters=1024, kernel_size=[7, 7], strides=1, 
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                padding="valid", activation=tf.nn.relu, use_bias=False)

            # Splitting into value and advantage stream
            valuestream, advantagestream = tf.split(conv4, 2, 3)
            valuestream = tf.layers.flatten(valuestream)
            advantagestream = tf.layers.flatten(advantagestream)
            advantage = tf.layers.dense(
                inputs=advantagestream, units=4,
                kernel_initializer=tf.variance_scaling_initializer(scale=2))
            value = tf.layers.dense(
                inputs=valuestream, units=1, 
                kernel_initializer=tf.variance_scaling_initializer(scale=2))

            # Combining value and advantage into Q-values
            q_values = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))
            best_action = tf.argmax(q_values, 1)


        return stateInput, q_values, best_action
