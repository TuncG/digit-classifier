from helpers import batch_accuracy, mnist_loss
from fastai.vision.all import *
from fastbook import *

class BasicOptim:
    def __init__(self,params,lr): 
        self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None


def train_epoch_optimizer(model, dl, opt):
    for xb,yb in dl:
        preds = model(xb)
        loss = mnist_loss(preds, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()


def validate_epoch_optimizer(model, valid_dl):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)