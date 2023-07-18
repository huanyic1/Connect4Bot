from connect4 import Connect4
from player import miniMaxBot, DeepQLearnBot
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
import copy

printGame = False
game = Connect4()
game.player1 = miniMaxBot(1, 12)
episodes = 100000
loss_fn = nn.MSELoss()
agent = DeepQLearnBot(2, not printGame)
agent.model.load()
optimizer = Adam(agent.model.parameters(), lr = 0.0001)
all_total_rewards = np.empty(episodes)
all_avg_rewards = np.empty(episodes)
batch_size = 40

miniMaxWins = 0
DeepQWins = 0
mistakes = 0
for e in range(episodes):
  done = False
  game.reset()
  total_rewards = 0
  place = game.player1.makeMove(game.board)
  game.make_move(place)
  if printGame:
    print(game)
  currStateBoard = copy.deepcopy(game.board)
  while not done:
    reward = 0
    action = agent.makeMove(game.board)
    err = game.make_move(action)
    if printGame:
        print(game)
    if err:
      reward = -50
      mistakes+=1
    else:
      check = game.check_win()
      if check == 2:
        done = True
        reward = 70
        DeepQWins+=1
      elif check == 3:
        done = True
        reward = 10
      else:
        p1Place = game.player1.makeMove(game.board)
        game.make_move(p1Place)
        if printGame:
            print(game)
        check = game.check_win()
        if check == 1:
          done = True
          miniMaxWins+=1
          reward = -30
        elif check ==3:
          done = True
          reward = 10
    nextStateBoard = copy.deepcopy(game.board)
    agent.model.addState(currStateBoard, action, nextStateBoard, reward, done)
    currStateBoard = nextStateBoard
    total_rewards+=reward
  # if len(agent.model.memory) > batch_size:
  agent.model.train(optimizer, loss_fn, batch_size)
  all_total_rewards[e] = total_rewards
  avg_reward = all_total_rewards[max(0, e - 100):e].mean()
  all_avg_rewards[e] = avg_reward
  if e % 100 == 0 :
      print("episode: {}/{}, average: {:.2f}".format(e, episodes, avg_reward))
      print("Minimax Wins: ", miniMaxWins, "Deep Q Wins: ", DeepQWins, "ERRORS: ", mistakes)
      miniMaxWins = 0
      DeepQWins = 0
      mistakes = 0
  if e % 1000 == 0 and e>0:
      agent.model.save()
      game.player1.seen = {}


