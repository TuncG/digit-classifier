from main import *
from helpers import *
from optimizer import *
from fastai.vision.all import *
from fastbook import *

def initialize_data():
    path = untar_data(URLs.MNIST)
    Path.BASE_PATH = path
    # digits = [ '0','1','2','3','4','5','6','7','8','9']
    digits = ['1','3','7']
    files_by_digit = store_files('training', path, digits)

    valid_by_digit = store_files('testing', path, digits)

    stacked_img_tensors = stack_tensors(files_by_digit)

    valid_stacked_img_tensors = stack_tensors(valid_by_digit)

    dset = generate_dset(stacked_img_tensors, files_by_digit, digits)
    
    valid_dset = generate_dset(valid_stacked_img_tensors, valid_by_digit, digits)
    return dset, valid_dset

def train_model(model, dl, opt, epochs, valid_dl):
    for i in range(epochs):
        train_epoch_optimizer(model, dl, opt)
        print(validate_epoch_optimizer(model, valid_dl), end=' ')

def main_optimized():
    
    dset, valid_dset = initialize_data()

    dl = DataLoader(dset, batch_size=256)
    valid_dl = DataLoader(valid_dset, batch_size=256)
    linear_model = nn.Linear(28*28, 3)  # 3 outputs for 3 classes
    lr = 0.1
    opt = BasicOptim(linear_model.parameters(), lr)

    print(validate_epoch_optimizer(linear_model, valid_dl))
    train_model(linear_model, dl, opt, 20, valid_dl)

    opt = SGD(linear_model.parameters(), lr)
    train_model(linear_model, dl, opt, 20, valid_dl)

if __name__ == "__main__":
    main_optimized()