import torch
import torch.nn as nn
import torch.optim as optim
from parse import parse

class Network(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )


    def forward(self, x):
        return self.net(x)
        
def train_model(file: str, input_cols: list[str], output_col: str, epochs: int = 100, lr: float = 0.01, tolerance: float = 2.0):
    all_cols = input_cols + [output_col]
    data = parse(file, all_cols)
    data = torch.tensor(data, dtype=torch.float32)

    X = data[:, :-1]
    y = data[:, -1].unsqueeze(1)

    model = Network(input_size = X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        correct = (torch.abs(output - y) <= tolerance).float().mean().item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Acc (±{tolerance}): {correct*100:.2f}%")
    
    return model