import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms
import numpy as np
import tensorflow as tf
class NNQ(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layers = [
            nn.Linear(in_features=42, out_features=64),
            nn.ReLU(), 
            nn.Linear(in_features=64, out_features=128, bias=True),
            nn.ReLU(), 
            nn.Linear(in_features=128, out_features=7, bias=True)
        ] 
        self.model = nn.Sequential(*layers).to(self.device)

        # initialize our optimizer. We'll use Adam
        self.optimizer = Adam(self.parameters(), lr=1e-3)
        self.load()
        self.lossFunc = nn.CrossEntropyLoss()
    
    def save(self): 
        torch.save(self.model.state_dict(), "NeuralNet.pt")
    
    def load(self): 
        try:
            self.model.load_state_dict(torch.load("NeuralNet.pt")).eval()
        except: 
            print("No model found, initializing blank")
    
    def train(self, board, reward): 
        flattenedBoard = board.flatten()
        tensor1 = tf.convert_to_tensor(flattenedBoard)
        predictedReward = self.model(tensor1)
        loss = self.lossFunc(predictedReward, reward)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def forward(self, board):
        return self.model(tf.convert_to_tensor(board.flatten()))

if __name__ == '__main__':
    neuralNet = NNQ()
    board = np.zeros((6,7))
    neuralNet.train(board, 0)
       
