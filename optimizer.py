from helpers import calc_grad

class BasicOptim:
    def __init__(self,params,lr): 
        self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None


def train_epoch(model, dl, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model, params[0], params[1])
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()
