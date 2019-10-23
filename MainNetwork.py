import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from LoadData import DataBatch
from Logging import LoggingHelper
from LoadCfg import NetManager
from Tensorboard import Mytensorboard
logger = LoggingHelper.get_instance().logger


from itertools import cycle

# LoadData에서 담아놓은 data load
class MyDataset(Dataset):
    def __init__(self, data_class):
        # Test일 땐 batch_size = 1
        # Training, Validation일 땐 batch_size = BATCH_SIZE
        self.data = data_class.data
        self.block_size = data_class.block_size

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

    def __len__(self):
        return len(self.data)


# Main Network 구조 (conv layer=2, fc_layer=2)
# in_layers   : Conv input layer 개수 ( (residual map + QP map) * mode num )
# num_classes : FC output class 개수
# group       : mode num
class ConvNet(nn.Module):
    def __init__(self, in_layers, num_classes, block_size, group_num):
        super().__init__()

        self.block_size = block_size

        self.conv1 = nn.Conv2d(in_channels=in_layers, out_channels=16 * group_num, kernel_size=3, stride=1, padding=1, groups=group_num)
        self.bn1 = nn.BatchNorm2d(16*group_num)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16 * group_num, out_channels=16 * group_num, kernel_size=3, stride=1, padding=1, groups=group_num)
        self.bn2 = nn.BatchNorm2d(16 * group_num)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # parameter 개수가 너무 많아서 속도 저하됨. parameter를 줄여주기 위한 conv. 값 변화는 없음
        self.conv3 = nn.Conv2d(in_channels=16 * group_num, out_channels=32, kernel_size=1, stride=1, padding=0, groups=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * (self.block_size//4) * (self.block_size//4), 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1

        for s in size:
            num_features *= s
        return num_features


if '__main__' == __name__:
    logger.info('Configs')
    for key,value in NetManager.cfg.member.items():
        logger.info('%s : %s' %(key,value))

    batch_size = NetManager.BATCH_SIZE
    train_data= DataBatch(istraining=NetManager.TRAINING, batch_size=batch_size, dataset_num=NetManager.DATASET_NUM)
    valid_data= DataBatch(istraining=NetManager.VALIDATION, batch_size=batch_size, dataset_num=NetManager.DATASET_NUM//8)
    test_data= DataBatch(istraining=NetManager.TEST, batch_size=1, dataset_num=NetManager.DATASET_NUM//8)

    # Training, Validation, Test data load 및 training 시작
    trainingDataset = MyDataset(train_data)
    validationDataset = MyDataset(valid_data)
    testDataset = MyDataset(test_data)

    trainingDataLoader = DataLoader(trainingDataset, batch_size=DataBatch.BATCH_SIZE, shuffle=True, drop_last=True,num_workers=DataBatch.NUM_WORKER)
    validationDataLoader = DataLoader(validationDataset, batch_size=DataBatch.BATCH_SIZE, shuffle=True, drop_last=True,num_workers=DataBatch.NUM_WORKER)
    testDataLoader = DataLoader(testDataset, batch_size=1, shuffle=True, num_workers=0)



    # network 객체 선언 및 구조 확인(summary)
    modeNum = 6
    blockSize = 32
    net = ConvNet(2 * modeNum, modeNum, trainingDataset.block_size, modeNum)
    summary(net, (2 * modeNum, trainingDataset.block_size, trainingDataset.block_size), device='cpu')

    train_iter = cycle(trainingDataLoader)
    valid_iter = cycle(validationDataLoader)
    test_iter = cycle(testDataLoader)



    # loss function : Cross Entropy
    # optimization  : Adam optimization
    # lr_scheduler  : learning rate 조정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=DataBatch.LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(DataBatch.OBJECT_EPOCH*0.5), int(DataBatch.OBJECT_EPOCH*0.75)], gamma=0.1, last_epoch=-1)


    # GPU 병렬로 사용할 지 단일로 사용할 지 처리하는 부분
    if DataBatch.IS_GPU:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
            else:
                net.cuda()

    # result_period : n번째마다 loss, accuracy 확인
    result_period = DataBatch.PRINT_PERIOD
    fBatch_size = float(DataBatch.BATCH_SIZE)
    tensorboard = Mytensorboard()

    logger.info('Training Set : %s %s' %(len(trainingDataLoader), train_data.each_data_num))
    logger.info('Validation Set : %s' %len(validationDataLoader))
    logger.info('Test set : %s' %len(testDataLoader))
    logger.info('Training Start')
    # EPOCH 수만큼 반복
    for epoch in range(DataBatch.OBJECT_EPOCH):
        train_loss = 0.0
        train_acc = 0.0

        # data 개수만큼 반복 (Training)
        for i in range(len(trainingDataLoader)):
            modes, residual = next(train_iter)
            if DataBatch.IS_GPU:
                modes = modes.cuda()
                residual = residual.cuda()
            # mode : prediction mode
            # residual : residual block value
            # output : network 학습 후 값 (softmax)
            t_mode = modes[:, 0]
            optimizer.zero_grad()
            t_output = net(residual)

            # running_loss : output과 실제 prediction mode 간의 cross entropy loss
            # running_acc  : 실제 prediction mode와 얼마나 일치하는가? (%)
            t_loss = criterion(t_output, t_mode)
            t_loss.backward()
            optimizer.step()

            t_prediction = t_output.data.max(1)[1]
            train_temp = t_prediction.eq(t_mode.data).sum().item() / fBatch_size * 100

            train_loss += t_loss.item()
            tensorboard.SetLoss('Accuracy', train_temp)
            tensorboard.plotScalars()
            tensorboard.step += 1
            train_acc += train_temp
            # 출력하고자 하는 period에 도달하면 loss, accuracy 출력
            if i % result_period == result_period - 1:
                logger.info("TRAINING [Epoch : %s] loss : %.4f, Accuracy : %.4f" % (epoch + 1, train_loss / result_period, train_acc / result_period))

                train_loss = 0.0
                train_acc = 0.0


        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0

            # data 개수만큼 반복 (Validation)
            for i in range(len(validationDataLoader)):
                modes, residual = next(valid_iter)
                if DataBatch.IS_GPU:
                    modes = modes.cuda()
                    residual = residual.cuda()

                v_mode = modes[:, 0]
                v_output = net(residual)

                v_loss = criterion(v_output, v_mode)
                v_prediction = v_output.data.max(1)[1]
                val_acc += v_prediction.eq(v_mode.data).sum().item() / fBatch_size * 100
                val_loss += v_loss.item()

            # 출력하고자 하는 period에 도달하면 loss, accuracy 출력
            logger.info("VALIDATION [Epoch : %s] loss : %.4f, Accuracy : %.4f" % (epoch + 1, val_loss / len(validationDataLoader), val_acc / len(validationDataLoader)))

        lr_scheduler.step()


    # data 개수만큼 반복 (Test)
    with torch.no_grad():
        test_loss = 0.0
        test_acc = 0.0

        for i in range(len(testDataLoader)):
            modes, residual = next(test_iter)
            if DataBatch.IS_GPU:
                modes = modes.cuda()
                residual = residual.cuda()

            # mode : prediction mode
            # residual : residual block value
            # output : network 학습 후 값 (softmax)
            mode = modes[:, 0]
            output = net(residual)

            # running_loss : output과 실제 prediction mode 간의 cross entropy loss
            # running_acc  : 실제 prediction mode와 얼마나 일치하는가? (%)
            loss = criterion(output, mode)
            prediction = output.data.max(1)[1]
            test_acc += prediction.eq(mode.data).sum().item() * 100.0
            test_loss += loss.item()

        logger.info("TEST [Epoch : %s] loss : %.4f, Accuracy : %.4f" % (1, test_loss/len(testDataLoader), test_acc/len(testDataLoader)))

    net.to(torch.device("cpu"))
    script_module = torch.jit.trace(net, torch.rand(1, 2 * modeNum, blockSize, blockSize))
    script_module.save(NetManager.MODEL_PATH + '/intra_model_32.pt')

    logger.info('Epoch %d Finished' %DataBatch.OBJECT_EPOCH)


