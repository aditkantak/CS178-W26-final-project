from torch import nn
import torch
from model_training import train, get_data
from torchvision.transforms import RandomCrop
    
class CNNv2 (nn.Module):
    def __init__(self):
        super().__init__()

        self.non_linearity = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.dropout2d = nn.Dropout2d(0.15)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), # 3, 32, 32 -> 16, 32, 32
            nn.BatchNorm2d(32),
            self.non_linearity,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 16, 32, 32 -> 16, 32, 32
            nn.BatchNorm2d(32),
            self.non_linearity,
            nn.MaxPool2d(kernel_size=2), # 16, 32, 32 -> 16, 16, 16
            self.dropout2d
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 16, 16, 16 -> 32, 16, 16
            nn.BatchNorm2d(64),
            self.non_linearity,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 32, 16, 16 -> 32, 16, 16
            nn.BatchNorm2d(64),
            self.non_linearity,
            nn.MaxPool2d(kernel_size=2), # 32, 16, 16 -> 32, 8, 8
            self.dropout2d
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 32, 8, 8 -> 64, 8, 8
            nn.BatchNorm2d(128),
            self.non_linearity,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 64, 8, 8 -> 64, 8, 8
            nn.BatchNorm2d(128),
            self.non_linearity,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 64, 8, 8 -> 64, 8, 8
            nn.BatchNorm2d(128),
            self.non_linearity,
            nn.MaxPool2d(kernel_size=2), # 64, 8, 8 -> 64, 4, 4
            self.dropout2d
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # 64, 4, 4 -> 128, 4, 4
            nn.BatchNorm2d(256),
            self.non_linearity,
            # nn.Conv2d(in_channels=200, out_channels=200, kernel_size=3, stride=1, padding=1), # 128, 4, 4 -> 128, 4, 4
            # nn.BatchNorm2d(200),
            # self.non_linearity,
            # nn.Conv2d(in_channels=200, out_channels=200, kernel_size=3, stride=1, padding=1), # 128, 4, 4 -> 128, 4, 4
            # nn.BatchNorm2d(200),
            # self.non_linearity,
            # nn.MaxPool2d(kernel_size=2), # 128, 4, 4 -> 128, 2, 2 #-----------> POTENTIAL CHANGE REMOVE THIS TO KEEP SPATIAL DIMENSION HIGHER <---------------
            self.dropout2d
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1) # 128, 4, 4 -> 128, 1, 1
        
        # flatten/squeeze 128, 1, 1 -> 128,

        self.head = nn.Sequential(
            # nn.Linear(in_features=256, out_features=200), # 128, -> 256,
            # self.non_linearity,
            # self.dropout,
            nn.Linear(in_features=256, out_features=10)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, X, y=None):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)

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
    torch.manual_seed(12345)

    train_data, val_data, test_data = get_data("standardize", random_crop=True, val_split=0.5)

    BATCH_SIZE = 250
    LR = 7e-3
    EPOCHS = 75
    EVAL_INTERVAL = 50
    GAMMA = 1
    OPTIM = "AdamW"
    WEIGHT_DECAY = 1e-4
    LR_SCHEDULER_TYPE = "Cosine"
    CHECKPOINT_THRESHOLD = 0.40

    cnnv2 = CNNv2()
    train(cnnv2, train_data, BATCH_SIZE, EPOCHS, LR, 
          val_data=val_data, 
          eval_interval=EVAL_INTERVAL, 
          optim=OPTIM, 
          lr_decay_factor=GAMMA, 
          l2_reg_strength=WEIGHT_DECAY, 
          checkpoint_threshold=CHECKPOINT_THRESHOLD
          )