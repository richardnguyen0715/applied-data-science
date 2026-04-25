"""Fraud detection model definitions"""

import torch
import torch.nn as nn


class FraudFeedforward(nn.Module):
    """Feedforward neural network for fraud detection"""
    
    def __init__(self, input_size: int = 29, hidden_layers: list = None, dropout: float = 0.3):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class FraudLSTM(nn.Module):
    """LSTM-based fraud detection model"""
    
    def __init__(self, input_size: int = 29, hidden_size: int = 64, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Reshape for LSTM if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return self.sigmoid(output)
