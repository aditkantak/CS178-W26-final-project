from torch import nn
import torch
from model_training import train, get_data
from torchvision.transforms import RandomCrop

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
    
class CNNv2 (nn.Module):
    def __init__(self):
        super().__init__()

        self.non_linearity = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout2d(0.2)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # 3, 32, 32 -> 16, 32, 32
            nn.BatchNorm2d(16),
            self.non_linearity,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), # 16, 32, 32 -> 16, 32, 32
            nn.BatchNorm2d(16),
            self.non_linearity,
            nn.MaxPool2d(kernel_size=2) # 16, 32, 32 -> 16, 16, 16
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # 16, 16, 16 -> 32, 16, 16
            nn.BatchNorm2d(32),
            self.non_linearity,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 32, 16, 16 -> 32, 16, 16
            nn.BatchNorm2d(32),
            self.non_linearity,
            nn.MaxPool2d(kernel_size=2) # 32, 16, 16 -> 32, 8, 8
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 32, 8, 8 -> 64, 8, 8
            nn.BatchNorm2d(64),
            self.non_linearity,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 64, 8, 8 -> 64, 8, 8
            nn.BatchNorm2d(64),
            self.non_linearity,
            nn.MaxPool2d(kernel_size=2) # 64, 8, 8 -> 64, 4, 4
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 64, 4, 4 -> 128, 4, 4
            nn.BatchNorm2d(128),
            self.non_linearity,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 128, 4, 4 -> 128, 4, 4
            nn.BatchNorm2d(128),
            self.non_linearity,
            nn.MaxPool2d(kernel_size=2) # 128, 4, 4 -> 128, 2, 2 #-----------> POTENTIAL CHANGE REMOVE THIS TO KEEP SPATIAL DIMENSION HIGHER <---------------
        )

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0) # 128, 4, 4 -> 128, 1, 1
        
        # flatten/squeeze 128, 1, 1 -> 128,

        self.head = nn.Linear(in_features=128, out_features=10) # 128, -> 10,

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, X, y=None):
        X = self.layer1(X)
        X = self.dropout(self.layer2(X))
        X = self.layer3(X)
        X = self.dropout(self.layer4(X))

        X = self.avgpool(X)

        X = torch.squeeze(X, dim=(2, 3))

        logits = self.head(X)

        if y is None:
            return logits, None
        else:
            loss = self.loss_fn(logits, y)
            return logits, loss
    
    def predict(self, X, k=0):
        all_logits = []
        
        all_logits.append(torch.squeeze(self(X)[0])) # shape 

        random_crop = RandomCrop(32, padding=4)
        for _ in range(k):
            cropped_X = random_crop(X)
            all_logits.append(torch.squeeze(self(cropped_X)[0]))

        all_logits = torch.stack(all_logits)

        final_logits = torch.mean(all_logits, dim=0)

        return torch.argmax(final_logits, dim=-1)
    
if __name__ == "__main__":
    vgglikecnn = VGGLikeCNN()
    cnnv2 = CNNv2()

    train_data, test_data = get_data("standardize")

    BATCH_SIZE = 500
    LR = 1e-1
    EPOCHS = 25
    EVAL_INTERVAL = 50

    train(vgglikecnn, train_data, BATCH_SIZE, EPOCHS, LR, val_data=test_data, eval_interval=EVAL_INTERVAL)
    train(cnnv2, train_data, BATCH_SIZE, EPOCHS, LR, val_data=test_data, eval_interval=EVAL_INTERVAL)