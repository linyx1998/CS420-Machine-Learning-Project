import numpy as np
import torch

from vgg16 import VGG16
from myRead import read_data
from run_vgg16 import  Ddataset


BATCH_SIZE=64
data_reader = read_data(isone=False)
x_train, x_test, x_valid, y_train, y_test, y_valid = data_reader.get_data()
test_set = Ddataset(x_test, y_test)
testLoader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)


# model = VGG16(n_classes=25)
model = torch.load('temp_models/epoch0.pth')
model.cuda()
# model.load_state_dict(torch.load('cnn.pkl'))


model.eval()
correct = 0
total = 0

for images, labels in testLoader:
    images = images.cuda()
    _, outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    # print(predicted, labels, correct, total)
    
print("avg acc: %f" % (100* correct/total))