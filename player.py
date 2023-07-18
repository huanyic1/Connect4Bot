import random
import json 
import numpy as np
import math
import copy
from NeuralNet import NNQ
import torch
# import main
# import mainWithMemo
import bitMask
import time

class Player():
    convert= {1: "O", 2: "X"}
    oneRewards = json.load(open('p1Rewards.json'))
    twoRewards = json.load(open('p2Rewards.json'))
 
    def __init__(self, turn): 
        self.turn = turn
        

    def makeMove(self, board): 
        return int(input(f'Please make move player {self.convert[self.turn]} \n'))
    
    # def saveRewards(self):
    #     if(self.turn == 1): 
    #         with open("p1rewards.json", "w") as outfile:
    #             json.dump(self.oneRewards, outfile)
    #     if(self.turn == 2):
    #         with open("p2rewards.json", "w") as outfile:
    #             json.dump(self.twoRewards, outfile)
    
    # def update(self, board, result): 
    #     boardStr = np.array2string(board)
    #     rewardsTable = self.oneRewards if self.turn == 1 else self.twoRewards
    #     rewardsFile = "p1rewards.json" if self.turn == 1 else "p2rewards.json"
    #     if boardStr not in rewardsTable: 
    #         rewardsTable[boardStr] = result
        
        # with open(rewardsFile, "w") as outfile:
        #     json.dump(rewardsTable, outfile)
        

class RandomBot(Player): # need to also be able to write to and read from weights file. 
    
    def makeMove(self, board): 
        choices = self.get_legal_spots(board)
        c= choices[random.randint(0, len(choices)-1)] #store states as a list of tuples. (move, board, score value)
        #print(c)
        return c
    
    def get_legal_spots(self, board):
        avail = []
        top_row = board[0]
        for i in range(7):
            if top_row[i] == 0: 
                avail.append(i)
        return avail

class DeepQLearnBot(RandomBot):
    def __init__(self, turn, training):
        self.turn = turn
        self.model = NNQ()
        self.training = training


    def makeMove(self, board):
        epsilon = self.model.epsilon if self.training else 0
        action_decision = random.choices(['model','random'], weights = [1 - epsilon, epsilon])[0]
        if action_decision == 'random':
            return super().makeMove(board)
        else:
            return torch.argmax(self.model.forward(board))

class miniMaxBot(Player):
    def __init__(self, turn, depth): 
        super().__init__(turn)
        self.depth = depth
        self.seen = {}

    def makeMove(self, board):
        _, place = bitMask.convertAndCall(board, self.depth, self.turn == 1, -20, 20, self.seen, self.turn)
        return place

    # def get_position_mask_bitmap(self, board, player):
        position, mask = '', ''
        # Start with right-most column
        for j in range(6, -1, -1):
            # Add 0-bits to sentinel 
            mask += '0'
            position += '0'
            # Start with bottom row
            for i in range(0, 6):
                #mask += ['0', '1'][board[i][j] != 0]
                mask += "1" if board[i][j] != 0 else "0"
                position += "1" if board[i][j] == player else "0"
                #position += ['0', '1'][board[i][j] == player]
        return int(position, 2), int(mask, 2)
    
    """
    Takes in board, returns (value, location)
    """
    # def minimax(self, board, depth, maximizingPlayer, alpha, beta, seen):
    #     discountFactor = 0.99 # for prioritizing faster wins, not sure if want to implement
    #     check = self.check_win(board)
    #     if depth == 0 or check:
    #         if check == 1: 
    #             return (10, None)
    #         elif check == 2:
    #             return (-10, None)
    #         else:
    #             return (0, None)
        
    #     children = [3, 4, 2, 5, 1, 6, 0] # naturally want to explore middle first
    #     if maximizingPlayer:
    #         maxEval = float("-inf")
    #         maxPlace = None 
    #         for child in children:
    #             if board[0][child]!=0: 
    #                 continue
    #             newBoard = copy.deepcopy(board)
    #             self.make_move(child, newBoard, 1)
    #             strBoard = str(newBoard)
    #             if strBoard in seen: 
    #                 eval = seen[strBoard]
    #             else:
    #                 eval, _ = self.minimax(newBoard, depth-1, False, alpha, beta, seen)
    #                 seen[strBoard] = eval
    #             if eval > maxEval:
    #                 maxEval = eval
    #                 maxPlace = child
    #             alpha = max(alpha, eval)
    #             if beta<=alpha:
    #                 break
    #         return (discountFactor*maxEval, maxPlace)
    #     else:
    #         minEval = float("inf")
    #         minPlace = None
    #         for child in children:
    #             if board[0][child]!=0: 
    #                 continue
    #             newBoard = copy.deepcopy(board)
    #             self.make_move(child, newBoard, 2)
    #             strBoard = str(newBoard)
    #             if strBoard in seen: 
    #                 eval = seen[strBoard]
    #             else:
    #                 eval, _ = self.minimax(newBoard, depth-1, True, alpha, beta, seen)
    #                 seen[strBoard] = eval
    #             if eval < minEval:
    #                 minEval = eval
    #                 minPlace = child
    #             beta = min(beta, eval)
    #             if beta<= alpha:
    #                 break
    #         return (discountFactor*minEval, minPlace)
        
    # def minimax2(self, board, depth, maximizingPlayer, alpha, beta):
    #     discountFactor = 0.99 # for prioritizing faster wins, not sure if want to implement
    #     check = self.check_win(board)
    #     if depth == 0 or check:
    #         if check == 1: 
    #             return (10, None)
    #         elif check == 2:
    #             return (-10, None)
    #         else:
    #             return (0, None)
        
    #     children = [3, 4, 2, 5, 1, 6, 0] # naturally want to explore middle first
    #     if maximizingPlayer:
    #         maxEval = float("-inf")
    #         maxPlace = None 
    #         for child in children:
    #             if board[0][child]!=0: 
    #                 continue
    #             newBoard = copy.deepcopy(board)
    #             self.make_move(child, newBoard, 1)
    #             eval, _ = self.minimax2(newBoard, depth-1, False, alpha, beta)
    #             if eval > maxEval:
    #                 maxEval = eval
    #                 maxPlace = child
    #             alpha = max(alpha, eval)
    #             if beta<=alpha:
    #                 break
    #         return (discountFactor*maxEval, maxPlace)
    #     else:
    #         minEval = float("inf")
    #         minPlace = None
    #         for child in children:
    #             if board[0][child]!=0: 
    #                 continue
    #             newBoard = copy.deepcopy(board)
    #             self.make_move(child, newBoard, 2)
    #             eval, _ = self.minimax2(newBoard, depth-1, True, alpha, beta)
    #             if eval < minEval:
    #                 minEval = eval
    #                 minPlace = child
    #             beta = min(beta, eval)
    #             if beta<= alpha:
    #                 break
    #         return (discountFactor*minEval, minPlace)
        
    # def check_win(self, board): 
    #     filled = 0
    #     for r in range(5, -1, -1): 
    #         for c in range(7): 
    #             if board[r][c]!= 0:
    #                 filled+=1
    #                 if r>2: # checks vertical
    #                     win = True
    #                     for i in range(1, 4): 
    #                         if board[r-i][c]!= board[r][c]: 
    #                             win = False
    #                             break
    #                     if win: 
    #                         return (board[r][c])
    #                 if c<4: # checks horizontal
    #                     win = True
    #                     for i in range(1, 4): 
    #                         if board[r][c+i]!= board[r][c]: 
    #                             win = False
    #                             break
    #                     if win: 
    #                         return (board[r][c])
    #                 if r>2 and c<4: # checks upward diagonal
    #                     win = True
    #                     for i in range(1, 4): 
    #                         if board[r-i][c+i]!= board[r][c]: 
    #                             win = False
    #                             break
    #                     if win: 
    #                         return (board[r][c])
    #                 if r<3 and c<4: # checks downward diagonal
    #                     win = True
    #                     for i in range(1, 4): 
    #                         if board[r+i][c+i]!= board[r][c]: 
    #                             win = False
    #                             break
    #                     if win: 
    #                         return (board[r][c])
    #     if filled == 42: 
    #         return 3
        
    # def make_move(self, place, board, turn): 
    #     if place > 6 or board[0][place]!= 0: 
    #         return -1
    #     for r in range(6): 
    #         if board[5-r][place] == 0: 
    #             if turn == 1: 
    #                 board[5-r][place] = 1 #O makes first move
    #                 turn +=1
    #             else: 
    #                 board[5-r][place] = 2
    #                 turn -=1
    #             last_move = (5-r, place)
    #             return 



    #     # start = time.time()
    #     # seen = {}
    #     # _, place = self.minimax(board, 5, self.turn == 1, float("-inf"), float("inf"), seen)
    #     # end = time.time()
    #     # print(place)
    #     # print("Time elapsed for python with memoization", end - start)

    #     # start = time.time()
    #     # _, place = self.minimax2(board, 5, self.turn == 1, float("-inf"), float("inf"))
    #     # end = time.time()
    #     # print(place)
    #     # print("Time elapsed for python without memoization", end - start)

    #     # start = time.time()
    #     # _, place = main.minimax(list(board.flatten()), 7, self.turn == 1, -20, 20)
    #     # end = time.time()
    #     # print(place)
    #     # print("Time elapsed for cython without memo", end-start)

    #     # seen2 = {}
    #     # start = time.time()
    #     # _, place = mainWithMemo.minimax(list(board.flatten()), 7, self.turn == 1, -20, 20, seen2)
    #     # end = time.time()
    #     # print(place)
    #     # print("Time elapsed for cython with memo", end-start)

    #     # seen3 = {}
    #     # start = time.time()
    #     # position, mask = self.get_position_mask_bitmap(board, 1)
    #     # _, place = bitMask.minimax(position, mask, 10, self.turn == 1, -20, 20, seen3)
    #     # end = time.time()
    #     # print(place)
    #     # print("Time elapsed for cython with bitmask", end-start)

    #     # seen4 = {}
    #     # start = time.time()
    #     # _, place = bitMask.convertAndCall(board, 10, self.turn == 1, -20, 20, seen4, self.turn)
    #     # end = time.time()
    #     # print(place)

    #     # print("Time elapsed for cython with bitmask convert", end-start)

# def useless():
#     testBoard = np.array([
#         [0,0,0,0,0,0,0], 
#         [0,0,0,2,0,0,0],
#         [0,0,1,1,0,0,0],
#         [0,0,2,1,0,0,0],
#         [0,0,2,2,1,0,0],
#         [0,0,2,1,1,2,0]
#     ])
#     agent = miniMaxBot(1)
#     print(agent.makeMove(testBoard))
#     pos, mask = agent.get_position_mask_bitmap(testBoard, 1)
#     print(pos, mask)
#     print(bitMask.convertAndCall(testBoard, 1))
