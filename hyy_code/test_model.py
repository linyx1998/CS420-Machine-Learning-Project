
import torch

# from vgg16 import VGG16
from myRead import read_data
from run_vgg16 import  Ddataset

from draw import mydraw


BATCH_SIZE=64


data_reader = read_data(isone=False)
x_train, x_test, x_valid, y_train, y_test, y_valid = data_reader.get_data()
test_set = Ddataset(x_test, y_test)
# test_set = Ddataset(x_train, y_train)
# test_set = Ddataset(x_valid, y_valid)

testLoader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)


accus = []
for i in range(10):
# model = VGG16(n_classes=25)
    if(i!=3):
        continue
    print('testing on model:', i)
    # model = torch.load('temp_models/vgg19_epoch%d'%i)
    model = torch.load('temp_models/epoch%d.pth'%i)
    model.cuda()
    # model.load_state_ dict(torch.load('cnn.pkl'))


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

    avgaccu = 100* correct/total
    accus.append(avgaccu)
print(accus)
# mydraw(range(10), accus)