import torch
from torch import nn

from model_training import train, get_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Create a confusion matrix
def make_confusion_matrix(model, data, batch_size=500):
    CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    all_preds = []
    all_labels = []

    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

    model.eval()
    device = next(model.parameters()).device

    for X, y in dataloader:
        X = X.to(device)
        preds = model.predict(X)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = (all_preds == all_labels).sum() / len(all_labels) * 100 #Find the accuracy of the data
    print(f'Accuracy: {accuracy:.2f}%')

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CIFAR10_CLASSES)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


class Logistic_Regression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, X, y=None):
        X = torch.flatten(X, start_dim=1) #flatten into b, 32*32*3
        
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
    OUTPUT_SIZE = 10
     # 10 possible outputs

    BATCH_SIZE = 250
    LR = 0.005 #0.05,0.01, 0.1, 1
    EPOCHS = 10
    EVAL_INTERVAL = 25

    model = Logistic_Regression(INPUT_SIZE, OUTPUT_SIZE)

    train_data, test_data = get_data("normalize")

    train(model, train_data, BATCH_SIZE, EPOCHS, LR, eval_interval=EVAL_INTERVAL, val_data=test_data)
    make_confusion_matrix(model, test_data)