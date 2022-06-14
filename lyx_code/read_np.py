import numpy as np
import os
from torch.utils.data import TensorDataset
import torch
import torch.utils.data
from torchvision import transforms

def read_data_np():
    data_folder = "../np_dataset/"
    data_package = os.listdir(data_folder)

    data_path = data_folder + data_package[0]
    data = np.load(data_path, allow_pickle=True, encoding='latin1')

    X_train_strokes = data['train'].astype(np.float64)
    X_valid_strokes = data['valid'].astype(np.float64)
    X_test_strokes = data['test'].astype(np.float64)

    y_train_strokes = np.full((len(X_train_strokes)), 0)
    y_valid_strokes = np.full((len(X_valid_strokes)), 0)
    y_test_strokes = np.full((len(X_test_strokes)), 0)

    for category in range(1, len(data_package)):
        print(data_package[category])
        
        data_path = data_folder + data_package[category]
        data = np.load(data_path, allow_pickle=True, encoding='latin1')

        X_train_strokes = np.concatenate((X_train_strokes, data['train']))
        X_valid_strokes = np.concatenate((X_valid_strokes, data['valid']))
        X_test_strokes = np.concatenate((X_test_strokes, data['test']))

        y_train_strokes = np.concatenate((y_train_strokes, np.full((len(data['train'])), category)))
        y_valid_strokes = np.concatenate((y_valid_strokes, np.full((len(data['valid'])), category)))
        y_test_strokes = np.concatenate((y_test_strokes, np.full((len(data['test'])), category)))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), (0.5,))
    ])

    X_train_ = transform(X_train_strokes[0]).reshape((1,1,28,28))
    X_valid_ = transform(X_valid_strokes[0]).reshape((1,1,28,28))
    X_test_ = transform(X_test_strokes[0]).reshape((1,1,28,28))

    for i in range(len(X_train_strokes)):
        X_train_ = np.concatenate((X_train_, transform(X_train_strokes[i]).reshape((1,1,28,28))))
    for i in range(len(X_valid_strokes)):
        X_valid_ = np.concatenate((X_valid_, transform(X_valid_strokes[i]).reshape((1,1,28,28))))
    for i in range(len(X_test_strokes)):
        X_test_ = np.concatenate((X_test_, transform(X_test_strokes[i]).reshape((1,1,28,28))))

    X_train = torch.from_numpy(X_train_).float()
    y_train = torch.from_numpy(y_train_strokes).long().squeeze()
    X_valid = torch.from_numpy(X_valid_strokes).float()
    y_valid = torch.from_numpy(y_valid_strokes).long().squeeze()
    X_test = torch.from_numpy(X_test_strokes).float()
    y_test = torch.from_numpy(y_test_strokes).long().squeeze()

    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)
    print(X_test.shape, y_test.shape)

    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)

    return train_dataset, valid_dataset, test_dataset

# read_data()