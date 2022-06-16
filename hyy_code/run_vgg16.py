import imp
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from myRead import read_data
from vgg16 import VGG16
from vgg16 import VGG19



class Ddataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()

        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len







if(__name__=='__main__'):
    # 超参 ##########################
    device = torch.device("cuda")
    BATCH_SIZE = 64
    EPOCH = 10
    LEARNING_RATE=0.001
    #################################
    data_reader = read_data(isone=False)
    x_train, x_test, x_valid, y_train, y_test, y_valid = data_reader.get_data()
    # print(x_train.shape)
    print(type(x_train[0][0][0][0]))

    train_set = Ddataset(x_train, y_train)
    test_set = Ddataset(x_test, y_test)
    valid_set = Ddataset(x_valid, y_valid)

    trainLoader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)


    # 模型
    model = VGG19(n_classes=25)
    model.to(device)
    # model.load_state_dict(torch.load('cnn.pkl'))
    # Loss, Optimizer & Scheduler
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,)

    # 训练
    start_time = time.time()
    for epoch in range(EPOCH):

        avg_loss = 0
        cnt = 0
        for images, labels in trainLoader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            _, outputs = model(images)
            loss = cost(outputs, labels.long())
            avg_loss += loss.item()
            cnt += 1
            print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.item(), avg_loss/cnt))
            loss.backward()
            optimizer.step()
        scheduler.step(avg_loss)
        torch.save(model, 'temp_models/vgg19_epoch%d'%(epoch))

    end_time = time.time()
    print('train time: ', start_time - end_time)

    # # Test the model

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

    # Save the Trained Model
    torch.save(model, 'cnn.pth')









    # # 一些超参数的设定
    # BATCH_SIZE = 1024
    # # LAM = 0.05
    # LAMS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1, 1.5, 2]
    # LAM = LAMS[0]
    # EPOCHS = [1,5,10,15,20,50,100]
    # EPOCH = 20
    # LEARNING_RATE = 0.0001
    # # SHUFFLE = True
    # NUMS = 128

    # START_WITH = 0

    # lsss1 = []
    # lsss2 = []
    # accus = []
    # for temp_lam in LAMS:
    #     print("starting test lam: ", temp_lam)
    #     accu, loss1, loss2 = main_train(BATCH_SIZE, temp_lam, EPOCH, LEARNING_RATE, NUMS, 0)
    #     lsss1.append(loss1)
    #     lsss2.append(loss2)
    #     accus.append(accu)

    # print(accus,lsss1,lsss2)
    # plt.plot(LAMS, lsss1, label="Label loss")
    # plt.plot(LAMS, lsss2, label="Domain loss")
    # plt.xlabel("Lambda")
    # plt.ylabel("Loss")
    # plt.show()
# # 训练
# def train(fold, model, device, train_loader, optimizer, epoch, criterion1, criterion2):
#     train_loss1 = 123
#     train_loss2 = 123
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         # print(labels) 
        
#         label_pred, domain_pred = model(inputs)
#         #print(label_pred.squeeze().shape, labels[:,0].shape)
#         loss1 = criterion1(label_pred, labels[:,0].long())
#         # print(label_pred,labels)
#         # print(domain_pred.shape)
#         loss2 = criterion2(domain_pred, labels[:,1].long())
#         train_loss1 = loss1.item()
#         train_loss2 = loss2.item()
#         # 打印训练过程信息并保存比较好的模型,loss1是label的，loss2是domain的
#         if(i == 0 and ((epoch)%5 == 0 or epoch == EPOCH-1)):
            
#             print("epoch:  ", epoch, "   loss1:",loss1.item(),"    loss2:", loss2.item())
#             # if(loss1.item() < 0.1 and loss2.item() > 1.1) or (loss1.item() < 0.3 and loss2.item() > 10):
#             #     torch.save(model, "temp_model/myModel_loss1_%.2f_loss2_%.2f__.pth"%( loss1.item(), loss2.item() ))

#             # print("saving time:", savetime1 - savetime)

#         optimizer.zero_grad()

#         loss1.backward(retain_graph=True)
#         loss2.backward()

#         optimizer.step()

#     return train_loss1, train_loss2

# # 测试，输出在测试集上的预测准确率
# def test(fold, model, device, test_loader):
#     # model = torch.load()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test_loader:
#             inputs, labels = data
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             labels = labels[:,0].long()
#             label_pred, domain_pred = model(inputs)
#             _, predicted = torch.max(label_pred.data, dim=1)
#             total+=labels.size(0)
#             correct+=(predicted == labels).sum().item()

#     accu = 100*correct/total
#     print('Accuracy on fold: %i is %.2d %%'%(fold, accu))

#     return accu





# def main_train(BATCH_SIZE, LAM, EPOCH, LEARNING_RATE, NUMS, START_WITH):
#     torch.backends.cudnn.benchmark = True
#     data_set = Ddataset('dataset/EEG_X.mat', 'dataset/EEG_Y.mat')
#     # data_loader = DataLoader(dataset=data_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=0)


#     start_time = time()
#     # 15折交叉验证
#     accu_test = []
#     accu_train = []
#     loss1_list = []
#     loss2_list = []
#     kfold = KFold(n_splits=15, shuffle=False)
#     for fold,(train_idx,test_idx) in enumerate(kfold.split(data_set)):
#         if(fold < START_WITH): ## 为了跳过某个训练集
#             continue
#         print('------------fold no---------{}----------------------'.format(fold))
#         train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
#         test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

#         train_loader = torch.utils.data.DataLoader(
#                             data_set, 
#                             batch_size=BATCH_SIZE, sampler=train_subsampler)
#         test_loader = torch.utils.data.DataLoader(
#                             data_set,
#                             batch_size=BATCH_SIZE, sampler=test_subsampler)

#         # model.apply(reset_weights)

#         model = Model(lam=LAM, nums=NUMS)
#         model.to(device)
#         criterion1 = torch.nn.CrossEntropyLoss()
#         criterion1 = criterion1.to(device)
#         criterion2 = torch.nn.CrossEntropyLoss()
#         criterion2 = criterion2.to(device)

#         optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
#         temp_accu_test = 0
#         temp_accu_train = 0
#         temp_loss1 = 0
#         temp_loss2 = 0
#         for epoch in range(EPOCH):
#             temp_loss1, temp_loss2 = train(fold, model, device, train_loader, optimizer, epoch, criterion1, criterion2)
#             if(epoch%5 == 0 or epoch == EPOCH-1):
#                 temp_accu_test = test(fold,model, device, test_loader)  # 验证集的准确率
#                 temp_accu_train = test(fold,model, device, train_loader)# 训练集准确率
#         accu_test.append(temp_accu_test)  
#         accu_train.append(temp_accu_train) 
#         loss1_list.append(temp_loss1)
#         loss2_list.append(temp_loss2)
    
#     end_time = time()   
#     # print("mean of loss1 and loss2: ", np.array(loss1_list).mean(), np.array(loss2_list).mean())
#     # plt.bar(range(len(accu_test)), accu_test)
#     # # plt.scatter(range(len(accu_train)), accu_train)
#     # plt.xlabel("Test on who")
#     # plt.ylabel("Accuracy")
#     # plt.title("learning rate: %f, grl_lambda: %f, batch_size: %d, epoch: %d, NUMS:%d"%(LEARNING_RATE, LAM, BATCH_SIZE, EPOCH,NUMS))
    

#     # # 测试集上的平均准确率， 15折交叉验证
#     test_accuracy = np.array(accu_test).mean()
#     # print(test_accuracy)
#     # plt.show()
    
    
#     print("using TIME", end_time - start_time)

#     return test_accuracy, np.array(loss1_list).mean(), np.array(loss2_list).mean()