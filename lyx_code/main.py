from read_np import *
from read_pic import *
from functions import *
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Function
from torch import nn, optim
from torchvision import models
import time
import sys

BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCH_NUM = 50

if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset = read_data_pic()
    train_data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, 
                        batch_size=BATCH_SIZE, num_workers=8)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, 
                        batch_size=BATCH_SIZE, num_workers=8)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, 
                        batch_size=BATCH_SIZE, num_workers=8)

    if sys.argv[1] == 50:
        classifier = models.resnet50(pretrained=False)
        classifier.load_state_dict(torch.load('../pretrain_models/resnet50-19c8e357.pth'))
    elif sys.argv[1] == 34:
        classifier = models.resnet34(pretrained=False)
        classifier.load_state_dict(torch.load('../pretrain_models/resnet34-333f7ec4.pth'))
    else:
        classifier = models.resnet18(pretrained=False)
        classifier.load_state_dict(torch.load('../pretrain_models/resnet18-5c106cde.pth'))

    # adjust the number of classes
    num_ftrs = classifier.fc.in_features
    classifier.fc = nn.Linear(num_ftrs, 25)

    train_begin = time.time()
    if sys.argv[2] == 'auto':
        auto_train(classifier, train_data_loader, valid_data_loader, EPOCH_NUM, LEARNING_RATE, sys.argv[1])
    else:
        train(classifier, train_data_loader, valid_data_loader, EPOCH_NUM, LEARNING_RATE, sys.argv[1])
    train_end = time.time()
    print("trainging time:", train_end - train_begin, "s")

    test(classifier, test_data_loader)







