import os
import glob
import numpy as np
from torch.utils.data import Dataset


class SceneflowDataset(Dataset):
    def __init__(self, npoints=2048, root='./data_preprocessing/data_processed_maxcut_35_both_mask_20k_2k', partition='train'):
        self.npoints = npoints
        self.partition = partition
        self.root = root
        if self.partition=='train':
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
        self.cache = {}
        self.cache_size = 30000
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d] # deal with NaN
        print(self.partition, ': ',len(self.datapath))

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['points1'].astype('float32')
                pos2 = data['points2'].astype('float32')
                color1 = data['color1'].astype('float32')
                color2 = data['color2'].astype('float32')
                flow = data['flow'].astype('float32')
                mask1 = data['valid_mask1']

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)

        if self.partition == 'train':
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            color1 = color1[sample_idx1, :]
            color2 = color2[sample_idx2, :]
            flow = flow[sample_idx1, :]
            mask1 = mask1[sample_idx1]
        else:
            pos1 = pos1[:self.npoints, :]
            pos2 = pos2[:self.npoints, :]
            color1 = color1[:self.npoints, :]
            color2 = color2[:self.npoints, :]
            flow = flow[:self.npoints, :]
            mask1 = mask1[:self.npoints]

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center
        return pos1, pos2, color1, color2, flow, mask1

    def __len__(self):
        return len(self.datapath)


class SceneflowDataset_kitti():
    def __init__(self, root='./kitti_rm_ground/', npoints=16384, train=True):
        self.npoints = npoints
        self.root = root
        self.train = train
        self.datapath = glob.glob(os.path.join(self.root, '*.npz'))
        if train:
            self.datapath = self.datapath[0:100]
        else:
            self.datapath = self.datapath[0:150]
        self.cache = {}
        self.cache_size = 30000

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1'].astype('float32')
                pos2 = data['pos2'].astype('float32')
                flow = data['gt'].astype('float32')

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, flow)

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        if n1 >= self.npoints:
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)), axis=-1)
        if n2 >= self.npoints:
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)), axis=-1)

        pos1 = pos1[sample_idx1, :]
        pos2 = pos2[sample_idx2, :]
        flow = flow[sample_idx1, :]

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center

        color1 = np.zeros([self.npoints, 3]).astype('float32')
        color2 = np.zeros([self.npoints, 3]).astype('float32')
        mask = np.ones([self.npoints]).astype('float32')

        return pos1, pos2, color1, color2, flow, mask

    def __len__(self):
        return len(self.datapath)
