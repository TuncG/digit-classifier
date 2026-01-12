from fastai.vision.all import *
from fastbook import *

def sigmoid(x): return 1/(1+torch.exp(-x))
#initalize random weights per pixel function
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()

def linear1(xb, weights, bias): return xb@weights + bias

def mnist_loss(predictions, targets):
    # cross entropy already combines softmax and nll_loss
    return F.cross_entropy(predictions, targets.squeeze())

def calc_grad(xb, yb, model, weights, bias):
    preds = model(xb, weights, bias)
    loss = mnist_loss(preds, yb)
    loss.backward()

def train_epoch_optimizer(model,dl, opt):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()


def batch_accuracy(preds, yb):
    probs = preds.softmax(dim=1)
    predicted_class = probs.argmax(dim=1)
    corrects = (predicted_class.unsqueeze(1) == yb)
    return corrects.float().mean()

def validate_epoch(model,weights, bias, valid_dl):
    accs = [batch_accuracy(model(xb, weights, bias), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)


