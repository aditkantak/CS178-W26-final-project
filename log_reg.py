import torch
from torch import nn

from model_training import train

class Logistic_Regression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, X, y=None):
        X = torch.flatten(X, start_dim=1) #flatten into 1 dimensional
        
        logits = self.linear(X) # return raw logits so that torch cross entropy loss works 
        
        if y is None: 
            return logits, None
        else:
            loss = self.loss_fn(logits, y)
            return logits, loss

    @torch.no_grad()
    def predict(self, X):
        X = torch.flatten(X, start_dim=1) #flatten input into b, 32*32*3
        
        logits, _ = self(X)

        probs = self.softmax(logits) #logit probabilities in shape of b, 10

        return torch.argmax(probs, dim=-1) #return probabilities in b, 1

if __name__ == "__main__":
    INPUT_SIZE = 32 * 32 * 3 # 32x32 images with 3 channels
    OUTPUT_SIZE = 10 # 10 possible outputs

    BATCH_SIZE = 500
    LR = 0.1
    EPOCHS = 5
    EVAL_INTERVAL = 10

    model = Logistic_Regression(INPUT_SIZE, OUTPUT_SIZE)

    train(model, BATCH_SIZE, EPOCHS, LR, EVAL_INTERVAL)


