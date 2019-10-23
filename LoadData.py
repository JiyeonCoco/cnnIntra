import os
import torch
import numpy as np
import random
import struct
from Logging import LoggingHelper
from LoadCfg import NetManager


class DataBatch(NetManager):
    # istraining -> ( 0: test / 1: training / 2: validation )
    def __init__(self, istraining, dataset_num, batch_size=1):
        random.seed(16)
        if istraining == NetManager.TRAINING:
            self.data_path = self.TRAINING_PATH
        elif istraining == NetManager.VALIDATION:
            self.data_path = self.VALIDATION_PATH
        else:
            self.data_path = self.TEST_PATH
        self.logger = LoggingHelper.get_instance().logger
        self.istraining = istraining
        self.learning_rate = self.LEARNING_RATE
        self.filelist = self.getFileList()
        self.filelist_num = len(self.filelist)
        self.dataset_num = dataset_num
        self.batch_size = batch_size
        self.data = []
        self.modes = []
        self.setDataset()



    # 디렉토리 내에 있는 bin file 차례대로 읽어서 구성 (sequences)
    def getFileList(self, pattern='.bin'):
        sequences = []

        for root, dirNames, fileNames in os.walk(self.data_path):
            for fileName in fileNames:
                if fileName.endswith(pattern):
                    sequences.append(os.path.join(root, fileName))

        return sequences

    # dataset을 batch size에 맞춰 구성
    # eachDataNum : 한 sequence 내에서 사용할 data의 개수
    def setDataset(self):
        self.each_data_num = self.dataset_num // self.filelist_num

        for file_list in self.filelist:
            self.unpackDataset(file_list, self.each_data_num)
        return

    # bin file로부터 data 읽어서 각각 구성 (intra mode, mode bit, residual)
    def unpackDataset(self, file_list, each_data_num):
        with open(file_list, 'rb') as data:
            file_size = os.path.getsize(file_list)
            QP = int(file_list.split('_')[-1].split('.')[0])
            self.block_size = int(file_list.split('_')[-2])
            area = self.block_size * self.block_size

            # block size: 8x8 -> 11 modes
            if self.block_size == self.PU_SIZE[0]:
                each_mode = self.MODE_NUM[0]
            # block size: 32x32 -> 6 modes
            elif self.block_size == self.PU_SIZE[1]:
                each_mode = self.MODE_NUM[1]

            # mode_size : intra prediction mode (unsigned char, 1 byte)
            # mode_bit_size : bits for signalling with lambda (float, 4 bytes * the number of modes)
            # residual_size : residual block (short, 2 bytes * area(8x8/32x32) * the number of modes)
            mode_size       = 1
            mode_bit_size   = 4 * each_mode
            residual_size   = area * each_mode
            input_data_size = mode_size + mode_bit_size + residual_size

            # data random shuffle
            random_arr = np.arange(0, file_size, input_data_size)
            np.random.seed(area)
            np.random.shuffle(random_arr)
            if each_data_num > len(random_arr):
                each_data_num = len(random_arr)
            random_arr = random_arr[:each_data_num]
            random_arr = np.sort(random_arr, axis=-1)
            random_arr -= np.array((0, *random_arr[:-1]))
            random_arr -= np.array([0] + [input_data_size]*(each_data_num-1))

            # unpack_str : modeBitSize + residualSize
            # QP_map : QP map (QP value / 52.0)
            unpack_str = str(each_mode) + 'f' + str(int(each_mode * area)) + 'b'

            # each_data_num : 한 sequence 내에서 사용할 data의 개수
            # mode          : n번째 input data의 intra prediction mode + mode bits
            # pixel         : n번째 input data의 residual data
            for i in range(each_data_num):
                data.seek(random_arr[i], 1)
                mode = [*struct.unpack('B', data.read(1))]
                info = np.array(struct.unpack(unpack_str, data.read(input_data_size - 1)), 'float32')
                pixel = []
                flag_cost = info[0:each_mode]
                resi = (info[each_mode:] / 255.0).reshape((6, 32, 32))

                for j in range(each_mode):
                    pixel.append(resi[j])
                    cost_map = np.zeros((self.block_size, self.block_size), dtype='float32') + (1000.0 / flag_cost[j])
                    pixel.append(cost_map)

                self.data.append((np.array(mode, dtype='long'), np.stack(pixel, axis=0)))