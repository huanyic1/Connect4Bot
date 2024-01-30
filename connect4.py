
import numpy as np
from player import Player
from player import RandomBot
from player import DeepQLearnBot
from player import miniMaxBot
import sys
import argparse
import termcolor
import json
import copy

class Connect4: 
    def __init__(self, player2):
        self.board = np.zeros((6,7))
        self.turn = 1 # flips between 1 and 2
        self.convert= {1: "O", 2: "X"}
        self.player1 = Player(1) #miniMaxBot(1, 10)
        if player2 == "Neural Net": 
            self.player2 = DeepQLearnBot(2, False)
            self.player2.model.load()
        elif player2 == "Minimax": 
            self.player2 = miniMaxBot(2, 12)
        else: 
            self.player2 = Player(2)
        self.last_move = None

    def reset(self):
      self.board = np.zeros((6,7))
      self.turn = 1
      self.last_move = None


    def __str__(self): 
        output = "-----------------------------\n"
        
        for r in range(len(self.board)): 
            output += "|"
            for c in range(len(self.board[0])): 
                item = self.board[r][c]
                if self.last_move and r == self.last_move[0] and c == self.last_move[1]:
                    if item == 1:
                        output += termcolor.colored(" O |", color="yellow", attrs = ["reverse"]) #highlights last move
                    else: 
                        output += termcolor.colored(" X |", color="yellow", attrs = ["reverse"])
                
                else:
                    if item == 0: 
                        output += "   |"
                    elif item == 1:
                        output += " O |"
                    else: 
                        output += " X |"

            output+="\n"
            output += "-----------------------------\n"
        output += "| 0 | 1 | 2 | 3 | 4 | 5 | 6 |"
        return output
    
    def make_move(self, place): 

        if place > 6 or self.board[0][place]!= 0: 
            return -1
        for r in range(6): 
            if self.board[5-r][place] == 0: 
                if self.turn == 1: 
                    self.board[5-r][place] = 1 #O makes first move
                    self.turn +=1
                else: 
                    self.board[5-r][place] = 2
                    self.turn -=1
                self.last_move = (5-r, place)
                return 
    
    def check_win(self): 
        filled = 0
        for r in range(5, -1, -1): 
            for c in range(7): 
                if self.board[r][c]!= 0:
                    filled+=1
                    if r>2: # checks vertical
                        win = True
                        for i in range(1, 4): 
                            if self.board[r-i][c]!= self.board[r][c]: 
                                win = False
                                break
                        if win: 
                            return (self.board[r][c])
                    if c<4: # checks horizontal
                        win = True
                        for i in range(1, 4): 
                            if self.board[r][c+i]!= self.board[r][c]: 
                                win = False
                                break
                        if win: 
                            return (self.board[r][c])
                    if r>2 and c<4: # checks upward diagonal
                        win = True
                        for i in range(1, 4): 
                            if self.board[r-i][c+i]!= self.board[r][c]: 
                                win = False
                                break
                        if win: 
                            return (self.board[r][c])
                    if r<3 and c<4: # checks downward diagonal
                        win = True
                        for i in range(1, 4): 
                            if self.board[r+i][c+i]!= self.board[r][c]: 
                                win = False
                                break
                        if win: 
                            return (self.board[r][c])
        if filled == 42: 
            return 3

def runGame(printGame = True, player2 = "Player"):
    game = Connect4(player2)
    if printGame: 
        print(game)
    while True: 
        if game.turn == 1:
            place = game.player1.makeMove(game.board)
        else: 
            place = game.player2.makeMove(game.board)
        ret = game.make_move(place)
        if ret: 
            print("ILLEGAL MOVE, MOVE SOMEWHERE ELSE\n")
            continue
        if printGame:
            print(game)
        check = game.check_win()
        if check==1: 
            if printGame:
                print(f"Player O won")
        elif check==2:
            if printGame:
                print("Player X won")
        elif check==3:
            if printGame:
                print("TIE")            
        if check:
            if printGame:
                print(game)
            return check
def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default = 1, type = int)
    parser.add_argument('--print', action='store_true') # when it is training and the two bots are playing each other, no need to print board everytime. 
    parser.add_argument('--simulate', dest='print', action='store_false')
    parser.add_argument('--player2')
    parser.set_defaults(print=True)
    arguments = parser.parse_args()
    player1Wins = 0
    player2Wins = 0
    ties = 0
    for _ in range(arguments.games):
       res = runGame(arguments.print, arguments.player2)
       if res == 1: player1Wins+=1
       if res == 2: player2Wins+=1
       if res == 3: ties+=1
    print(f"Results: \n Player1: {player1Wins} \n Player2: {player2Wins} \n Ties: {ties}")
        
if __name__ == '__main__':
    main(sys.argv[1:])
