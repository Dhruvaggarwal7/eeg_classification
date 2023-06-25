import torch
from models.basic_net import NNet
from dataset.eeg_dataset import training_eeg_loader

from utils.trainer import start_training

if __name__== '__main__':

        model = NNet()
        loader = training_eeg_loader(512)
        start_training(model, loader)
        torch.save(model.state_dict(), "nnet.pt")