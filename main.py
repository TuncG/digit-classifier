from fastai.vision.all import *
from fastbook import *
from helpers import *
import torch.nn.functional as F

def store_files(files, path, digits):
    files_by_digit = {}
    for d in digits:
        files_by_digit[d] = (path/files/d).ls().sorted()
    
    return files_by_digit

def stack_tensors(image_files):
    stacked_img_tensors = []

    for d, files in image_files.items():
        img_tensor = [tensor(Image.open(o)) for o in files]
        stacked_img_tensors.append(torch.stack(img_tensor).float()/255)
        print(d, " tensor stack completed")
    
    return stacked_img_tensors

def generate_dset(stacked_img_tensors, files_by_digit, digits):
    train_x = torch.cat(stacked_img_tensors).view(-1, 28*28)
    train_y = tensor(sum([[i]*len(files_by_digit[digits[i]]) for i in range(len(digits))], [])).unsqueeze(1)
    dset = list(zip(train_x,train_y))
    return dset


def main():
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

    weights = init_params((28*28,3))

    bias = init_params(3)

    dl = DataLoader(dset, batch_size=256)
    valid_dl = DataLoader(valid_dset, batch_size=256)
    lr = 0.1
    
    for i in range(20):
        train_epoch(linear1, dl, lr, [weights, bias])
        print(validate_epoch(linear1, weights, bias, valid_dl), end=' ')

if __name__ == "__main__":
    main()