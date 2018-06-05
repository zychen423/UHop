import torch
import os

def find_save_dir(model_name):
    counter = 0
    save_dir = f'../save_model/{model_name}_{counter}'
    while os.path.exists(save_dir):
        counter += 1
        save_dir = f'../save_model/{model_name}_{counter}'
    os.mkdir(save_dir)
    print(f'save_dir is {save_dir}')
    return save_dir

def save_model(model, path):
    path = os.path.join(path, 'model.pth')
    torch.save(model.state_dict(), path)
    print(f'save model at {path}')

def save_model_with_result(model, loss, acc, path):
    path = os.path.join(path, f'model_{loss:.4f}_{acc:.4f}.pth')
    torch.save(model.state_dict(), path)
    print(f'save model at {path}')

def load_model(model, path):
    path = os.path.join(path, 'model.pth')
    print(f'load model from: {path}')
    model.load_state_dict(torch.load(path))
    return model


