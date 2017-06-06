# -----------------------------
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
import keras
import h5py
import json
from keras.models import Sequential , load_model,model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.models import model_from_config
from util import clone_model
from keras.layers.advanced_activations import LeakyReLU, PReLU
import os
# Hyper Parameters:
FRAME_PER_ACTION = 85
GAMMA = 0.5 # decay rate of past observations
OBSERVE = 100. # timesteps to observe before training
EXPLORE = 50000. # frames over which to anneal epsilon
INITIAL_EPSILON = 0.1#0.01 # starting value of epsilon
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
    if os.path.exists("model.h5"):
      model = load_model("./model.h5")
    else:
      model = Sequential()
      model.add(Convolution2D(32, 8, 8, subsample=(4, 4),input_shape=(80,80,4)))  #80*80*4
      model.add(Activation('relu'))
      model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
      model.add(Activation('relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
      model.add(Activation('relu'))
      model.add(Flatten())
      model.add(Dense(512))
      model.add(Activation('relu'))
      model.add(Dense(self.actions))
      model.compile(loss='mean_squared_logarithmic_error',
                   optimizer=keras.optimizers.Adam(lr=1e-6),
                   metrics=['accuracy'])
      #model.compile(loss=keras.losses.categorical_crossentropy,
      #             optimizer=keras.optimizers.Adadelta(),
      #             metrics=['accuracy'])
    return model



  def copyTargetQNetwork(self):
    print('copy')
    self.copy_model = clone_model(self.model)
    #self.copy_model = copy.copy(self.model)

      
  def trainQNetwork(self):
      # Step 1: obtain random minibatch from replay memory
      minibatch = random.sample(self.replayMemory, BATCH_SIZE)
      state_batch = [data[0] for data in minibatch]
      action_batch = [data[1] for data in minibatch]
      reward_batch = [data[2] for data in minibatch]
      nextState_batch = [data[3] for data in minibatch]
      # Step 2: calculate y
      y_batch = self.model.predict(np.array(state_batch).reshape(-1,80,80,4))

      QValue_batch = self.model.predict(np.array(nextState_batch).reshape(-1,80,80,4))
      
      for i in range(0, BATCH_SIZE):
          terminal = minibatch[i][4]
          action_t = minibatch[i][1]
          if terminal:
              y_batch[i,np.argmax(action_t)] = reward_batch[i]
          else:
              y_batch[i,np.argmax(action_t)] = reward_batch[i] + (GAMMA * QValue_batch[i][np.argmax(action_t)])
      self.model.fit(np.array(state_batch).reshape(-1,80,80,4),y_batch,batch_size=32, epochs=10, verbose=0)
      
      if self.timeStep % UPDATE_TIME == 0:
        self.copyTargetQNetwork()

  def setPerception(self, nextObservation, action, reward, terminal):
    # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)a
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
    if self.timeStep % 2000 == 0:
      print("Now we save model")
      self.model.save("model.h5", overwrite=True)



  def getAction(self):
    current = self.currentState.reshape(1,80,80,4)
    QValue = self.copy_model.predict(current)[0]
    print(QValue)
    action = np.zeros(self.actions)
    action_index = 0
    if self.timeStep - FRAME_PER_ACTION > 0:
      if random.random() <= self.epsilon:
        action_index = random.randrange(self.actions)
        action[action_index] = 1
        print('random')
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

