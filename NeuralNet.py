import os
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import random
import copy
from math import exp, log
import time

class NNQ(nn.Module): # utilizing nn.Module
    def __init__(self):
        super(NNQ, self).__init__()
        episodes = 100000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        self.memory = []
        self.epsilon = .2
        self.gamma = 0.9 # discount rate
        self.epsilon_min = 0.01
        self.epsilon_decay = exp((log(self.epsilon_min) - log(self.epsilon))/(0.8*episodes))


    def save(self):
      try:
        torch.save(self.model.state_dict(), 'FinalConnect4Weights.pth')
        print("Model Saved successfully")
      except Exception as err:
        print("Couldn't save model")
        print(err)

    def load(self):
        try:
            content = torch.load('FinalConnect4Weights.pth')
            self.model.load_state_dict(content)
            print("model loaded successfully")
        except Exception as err:
            print("No model found, initializing blank")
            print(err)

    def clear(self):
        self.memory = []

    def addState(self, state, action, nextState, reward, done):
        self.memory.append((state, action, nextState, reward, done))

    def forward(self, board):
        return self.model(torch.from_numpy(board.flatten()).float())

    def train(self, optimizer, loss_fn, batchSize):
        #minibatch = random.sample(self.memory, batchSize)
        finalReward = 0
        counter = 0
        discountFactor = .8
        while self.memory:
          state, action, nextState, reward, done = self.memory.pop()
          if counter == 0:
            finalReward = reward
          if reward == 0:
            reward = discountFactor**counter * finalReward
        # for state, action, nextState, done in self.memory:
        #   target = reward
        #   if not done:
        #       target = reward + self.gamma * torch.max(self.forward(nextState)).item()
          predicted = self.forward(state)
          actual = torch.clone(predicted)
          actual[action] = actual[action] * (1-self.gamma) + reward * self.gamma
          loss = loss_fn(predicted, actual)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          counter+=1
        if self.epsilon > self.epsilon_min:
          self.epsilon *= self.epsilon_decay
        # self.memory = []






