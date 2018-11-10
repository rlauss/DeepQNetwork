import gym
import gym.spaces
from gym import wrappers
import numpy as np
import random
import os
import math
import copy
import datetime
from statistics import median, mean
from collections import Counter, deque
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from QAgent import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_episodes', 20000, 'Number of episodes.')
flags.DEFINE_string("mode", "train", "<train>, <test>.")
flags.DEFINE_boolean("render", False, "Enable game rendering.")
flags.DEFINE_boolean("monitor", False, "Enable game monitoring.")

class ProcessFrame:
    """Resizes and converts RGB Atari frames to grayscale"""
    def __init__(self, frame_height=84, frame_width=84):
        """
        Args:
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.crop_to_bounding_box(self.processed, 34, 0, 160, 160)
        self.processed = tf.image.resize_images(self.processed, 
                                                [self.frame_height, self.frame_width], 
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    def process(self, session, frame):
        """
        Args:
            session: A Tensorflow session object
            frame: A (210, 160, 3) frame of an Atari game in RGB
        Returns:
            A processed (84, 84, 1) frame in grayscale
        """
        return session.run(self.processed, feed_dict={self.frame:frame})

class Atari:
    """Wrapper for the environment provided by gym"""
    def __init__(self, envName, no_op_steps=10, agent_history_length=4):
        self.env = gym.make(envName)
        self.frame_processor = ProcessFrame()
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length
        if FLAGS.monitor:
            self.env = wrappers.Monitor(self.env, f'video/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    def reset(self, sess, evaluation=False):
        """
        Args:
            sess: A Tensorflow session object
            evaluation: A boolean saying whether the agent is evaluating or training
        Resets the environment and stacks four frames ontop of each other to 
        create the first state
        """
        frame = self.env.reset()
        self.last_lives = 0
        terminal_life_lost = True # Set to true so that the agent starts 
                                  # with a 'FIRE' action when evaluating
        if evaluation:
            for _ in range(random.randint(1, self.no_op_steps)):
                frame, _, _, _ = self.env.step(1) # Action 'Fire'
        processed_frame = self.frame_processor.process(sess, frame)   # (???)
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=2)
        #return terminal_life_lost
        return self.state

    def step(self, sess, action):
        """
        Args:
            sess: A Tensorflow session object
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal, info = self.env.step(action)  # (5?)
            
        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']
        
        processed_new_frame = self.frame_processor.process(sess, new_frame)   # (6?)
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2) # (6?)   
        self.state = new_state
        
        return processed_new_frame, reward, terminal, terminal_life_lost, self.state

def pre_process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[20:195, :]
    image = cv2.resize(image, (84, 84))
    #image = image / 255.0
    ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    #image = image.reshape((84, 84, 1))
    #plt.imshow(image)
    #plt.show()
    return image

def main(argv=None):
    # create enviroment
    atari = Atari('BreakoutDeterministic-v4')

    with tf.Session() as sess:
        agent = QAgent(sess)

        for i in range(FLAGS.num_episodes):
            done = False
            state = atari.reset(sess)
            total_reward = 0

            while not done:
                if FLAGS.render:
                   env.render()

                # training mode
                if FLAGS.mode == "train":
                    # get action
                    action = agent.get_action(state)
                    
                    # take action
                    next_frame, reward, done, terminal_life_lost, next_state = atari.step(sess, action)

                    # agent process
                    agent.process(next_frame, action, reward, done, terminal_life_lost)

                    #
                    state = next_state
                # evaluation mode
                elif FLAGS.mode == "test":
                    # get action
                    action = agent.get_action(state, training=False)

                    # take action
                    _, reward, done, _, state = atari.step(sess, action)
                    total_reward += reward

                    if done:
                        print(f'Episode: {i}, Reward: {total_reward}')
                else:
                    print('Invalid mode')
                    return

if __name__ == "__main__":
    tf.app.run()
    