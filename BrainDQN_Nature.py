# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import tensorflow as tf 
import numpy as np 
import random
from collections import deque 
import copy
import tensorflow as tf
import numpy as np
import random
from collections import deque
import keras
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D


# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100. # timesteps to observe before training
EXPLORE = 200000. # frames over which to anneal epsilon
INITIAL_EPSILON = 0.01#0.01 # starting value of epsilon
FINAL_EPSILON = 0.001#0.001 # final value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 100

class BrainDQN:

  def __init__(self,actions):
    # init replay memory
    self.replayMemory = deque()
    # init some parameters
    self.timeStep = 0
    self.epsilon = INITIAL_EPSILON
    self.actions = actions
    self.model = self.createQNetwork()
    self.copy_model = copy.copy(self.model)
    
  def createQNetwork(self):
    model = Sequential()
    model.add(Conv2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(80,80,4)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Conv2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(self.actions))
    model.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adadelta(),
                 metrics=['accuracy'])
    return model



  def copyTargetQNetwork(self):
    print('copy')
    self.copy_model = copy.copy(self.model)

  def createTrainingMethod(self):
    self.actionInput = tf.placeholder("float",[None,self.actions])
    self.yInput = tf.placeholder("float", [None]) 
    Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices = 1)
    self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
    self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


      
  def trainQNetwork(self):
      # Step 1: obtain random minibatch from replay memory
      minibatch = random.sample(self.replayMemory, BATCH_SIZE)
      state_batch = [data[0] for data in minibatch]
      action_batch = [data[1] for data in minibatch]
      reward_batch = [data[2] for data in minibatch]
      nextState_batch = [data[3] for data in minibatch]

      # Step 2: calculate y
      y_batch = []
      QValue_batch = self.model.predict(np.array(nextState_batch))
      for i in range(0, BATCH_SIZE):
          terminal = minibatch[i][4]
          if terminal:
              y_batch.append(reward_batch[i])
          else:
              y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

  
      if self.timeStep % UPDATE_TIME == 0:
        self.copyTargetQNetwork()

  def setPerception(self, nextObservation, action, reward, terminal):
    # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
    newState = np.append(
        self.currentState[:, :, 1:], nextObservation, axis=2)
    self.replayMemory.append(
        (self.currentState, action, reward, newState, terminal))
    if len(self.replayMemory) > REPLAY_MEMORY:
        self.replayMemory.popleft()
    if self.timeStep > OBSERVE:
        # Train the network
        self.trainQNetwork()
    state = ""
    if self.timeStep <= OBSERVE:
      state = "observe"
    elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
      state = "explore"
    else:
      state = "train"

    print("TIMESTEP", self.timeStep, "/ STATE", state, \
            "/ EPSILON", self.epsilon)

    self.currentState = newState
    self.timeStep += 1


  def getAction(self):
    current = self.currentState.reshape(1,80,80,4)
    QValue = self.copy_model.predict(current)
    action = np.zeros(self.actions)
    action_index = 0
    if self.timeStep % FRAME_PER_ACTION == 0:
      if random.random() <= self.epsilon:
        action_index = random.randrange(self.actions)
        action[action_index] = 1
      else:
        action_index = np.argmax(QValue)
        action[action_index] = 1
    else:
      action[0] = 1 # do nothing

    # change episilon
    if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
      self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

    return action

  def setInitState(self,observation):
    self.currentState = np.stack((observation, observation, observation, observation), axis = 2)
