from torch import nn
import torch
from model_training import train, get_data

class VGGLikeCNN (nn.Module):
    def __init__(self):
        super().__init__()

        self.non_linearity = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 3, 32, 32 -> 16, 32, 32
            self.non_linearity,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), # 16, 32, 32 -> 16, 32, 32
            self.non_linearity,
            nn.MaxPool2d(kernel_size=2) # 16, 32, 32 -> 16, 16, 16
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 16, 16, 16 -> 32, 16, 16
            self.non_linearity,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 32, 16, 16 -> 32, 16, 16
            self.non_linearity,
            nn.MaxPool2d(kernel_size=2) # 32, 16, 16 -> 32, 8, 8
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 32, 8, 8 -> 64, 8, 8
            self.non_linearity,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 64, 8, 8 -> 64, 8, 8
            self.non_linearity,
            nn.MaxPool2d(kernel_size=2) # 64, 8, 8 -> 64, 4, 4
        )

        # flatten 64, 4, 4 -> 1024,

        self.fclayer = nn.Sequential(
            nn.Linear(in_features=4*4*64, out_features=128), # 1024, -> 128,
            self.non_linearity,
            nn.Linear(in_features=128, out_features=128), # 128, -> 128,
            self.non_linearity,
            nn.Linear(in_features=128, out_features=10) # 128, -> 10,
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, X, y=None):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)

        X = torch.flatten(X, start_dim=1)

        logits = self.fclayer(X)

        if y is None:
            return logits, None
        else:
            loss = self.loss_fn(logits, y)
            return logits, loss
        
    def predict(self, X):
        logits, _ = self(X)

        preds = torch.argmax(logits, dim=-1)

        return preds
    
if __name__ == "__main__":
    model = VGGLikeCNN()

    train_data, test_data = get_data("standardize")

    BATCH_SIZE = 500
    LR = 1e-1
    EPOCHS = 20
    EVAL_INTERVAL = 10

    train(model, train_data, BATCH_SIZE, EPOCHS, LR, val_data=test_data)