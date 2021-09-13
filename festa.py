import torch.nn as nn
import torch
import torch.nn.functional as F
from layer import TemporalAbstraction_Attention, SpatialAbstraction_Attention, SpatialAbstraction,flow_fusion
from util import PointNetFeaturePropogation, PointNetSetUpConv


class FESTA(nn.Module):
    def __init__(self,args):
        super(FESTA,self).__init__()
        self.batch_size = args.batch_size
        self.bn = args.bn
        if args.rgb:
            self.sa1 = SpatialAbstraction(args, npoint=1024, radius=1.0, nsample=32, in_channel=3, mlp=[32,32,64], group_all=False)
        else:
            self.sa1 = SpatialAbstraction(args, npoint=1024, radius=1.0, nsample=32, in_channel=0, mlp=[32,32,64], group_all=False)

        self.sa2 = SpatialAbstraction_Attention(args, npoint=256, radius=2.0, nsample=32, in_channel=64, mlp=[64, 64, 128], group_all=False)
        self.sa3 = SpatialAbstraction(args, npoint=64, radius=2.0, nsample=8, in_channel=128, mlp=[128, 128, 256], group_all=False)
        self.sa4 = SpatialAbstraction(args, npoint=16, radius=4.0, nsample=8, in_channel=256, mlp=[256,256,512], group_all=False)

        self.fe_layer = TemporalAbstraction_Attention(args, radius=10.0, nsample=64, in_channel = 128, mlp=[128, 128, 128], pooling='max', corr_func='concat')
        self.fusion = flow_fusion(radius=2.0, nsample=32)

        self.su1 = PointNetSetUpConv(nsample=8, radius=2.4, f1_channel = 256, f2_channel = 512, mlp=[], mlp2=[256, 256])
        self.su2 = PointNetSetUpConv(nsample=8, radius=1.2, f1_channel = 128+128, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        self.su3 = PointNetSetUpConv(nsample=8, radius=0.6, f1_channel = 64, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        if args.rgb:
            self.fp = PointNetFeaturePropogation(in_channel = 256+3, mlp = [256, 256])
        else:
            self.fp = PointNetFeaturePropogation(in_channel = 256, mlp = [256, 256])

        self.conv1 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2=nn.Conv1d(128, 3, kernel_size=1, bias=True)

        self.conv3 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv4=nn.Conv1d(128, 1, kernel_size=1, bias=True)


    def forward(self, pc1, pc2, feature1, feature2):

        l1_pc1_fps, l1_pc1, l1_feature1, fps_index_l1_pc1 = self.sa1(pc1, feature1)
        l2_pc1_fps, l2_pc1, l2_feature1, grouped_index_l2_pc1 = self.sa2(l1_pc1, l1_feature1)

        l1_pc2_fps, l1_pc2, l1_feature2, fps_index_l1_pc2 = self.sa1(pc2, feature2)
        l2_pc2_fps, l2_pc2, l2_feature2, _ = self.sa2(l1_pc2, l1_feature2)

        _, l2_feature1_new = self.fe_layer(l2_pc1, None, l2_pc2, l2_feature1, l2_feature2, 64)
        _, l3_pc1, l3_feature1, fps_index_l3_pc1 = self.sa3(l2_pc1, l2_feature1_new)
        _, l4_pc1, l4_feature1, fps_index_l3_pc1 = self.sa4(l3_pc1, l3_feature1)

        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)

        if self.bn:
            x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        else:
            x = F.relu(self.conv1(l0_fnew1))
        sf_1 = self.conv2(x)
        if self.bn:
            y = F.relu(self.bn2(self.conv3(l0_fnew1)))
        else:
            y = F.relu(self.conv3(l0_fnew1))
        mask = self.conv4(y)
        mask = mask.squeeze(1)

        center_flow = self.fusion(l2_pc1.transpose(2, 1).contiguous(), pc1.transpose(2, 1).contiguous(), sf_1)
        l2_pc1_re = center_flow + l2_pc1

        # ............................... Fusion goes here ...................................
        _, l2_feature1_new_re = self.fe_layer(l2_pc1, l2_pc1_re, l2_pc2, l2_feature1, l2_feature2, 64)
        _, l3_pc1_re, l3_feature1_re, fps_index_l3_pc1_re = self.sa3(l2_pc1, l2_feature1_new_re)
        _, l4_pc1_re, l4_feature1_re, fps_index_l3_pc1_re = self.sa4(l3_pc1_re, l3_feature1_re)

        l3_fnew1_re = self.su1(l3_pc1_re, l4_pc1_re, l3_feature1_re, l4_feature1_re)
        l2_fnew1_re = self.su2(l2_pc1, l3_pc1_re, torch.cat([l2_feature1, l2_feature1_new_re], dim=1), l3_fnew1_re)
        l1_fnew1_re = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1_re)
        l0_fnew1_re = self.fp(pc1, l1_pc1, feature1, l1_fnew1_re)

        if self.bn:
            x_re = F.relu(self.bn1(self.conv1(l0_fnew1_re)))
        else:
            x_re = F.relu(self.conv1(l0_fnew1_re))
        sf_2 = self.conv2(x_re)

        if self.bn:
            y_re = F.relu(self.bn2(self.conv3(l0_fnew1_re)))
        else:
            y_re = F.relu(self.conv3(l0_fnew1_re))
        mask_2 = self.conv4(y_re)
        mask_2 = mask_2.squeeze(1)
        return sf_1, sf_2, mask, mask_2, l1_pc1_fps, l1_pc1, l2_pc1_fps, l2_pc1, l1_pc2_fps, l1_pc2, l2_pc2_fps, l2_pc2, l3_pc1, l4_pc1


class FESTA_Kitti(nn.Module):
    def __init__(self,args):
        super(FESTA_Kitti,self).__init__()
        self.batch_size = args.batch_size
        self.bn = args.bn

        if args.rgb:
            self.sa1 = SpatialAbstraction(args, npoint=4096, radius=1.0, nsample=256, in_channel=3, mlp=[32,32,64], group_all=False)
        else:
            self.sa1 = SpatialAbstraction(args, npoint=4096, radius=1.0, nsample=256, in_channel=0, mlp=[32,32,64], group_all=False)
        self.sa2 = SpatialAbstraction_Attention(args, npoint=1024, radius=2.0, nsample=256, in_channel=64, mlp=[64, 64, 128], group_all=False)
        self.sa3 = SpatialAbstraction(args, npoint=256, radius=2.0, nsample=32, in_channel=128, mlp=[128, 128, 256], group_all=False)
        self.sa4 = SpatialAbstraction(args, npoint=128, radius=4.0, nsample=32, in_channel=256, mlp=[256,256,512], group_all=False)
        self.fusion = flow_fusion(radius=2.0, nsample=128)
        self.fe_layer = TemporalAbstraction_Attention(args, radius=3.0, nsample=128, in_channel = 128, mlp=[128, 128, 128], pooling='max', corr_func='concat')

        self.su1 = PointNetSetUpConv(nsample=4, radius=2.4, f1_channel = 256, f2_channel = 512, mlp=[], mlp2=[256, 256])
        self.su2 = PointNetSetUpConv(nsample=4, radius=1.2, f1_channel = 128+128, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])
        self.su3 = PointNetSetUpConv(nsample=4, radius=0.6, f1_channel = 64, f2_channel = 256, mlp=[128, 128, 256], mlp2=[256])

        if args.rgb:
            self.fp = PointNetFeaturePropogation(in_channel = 256+3, mlp = [256, 256])
        else:
            self.fp = PointNetFeaturePropogation(in_channel = 256, mlp = [256, 256])

        self.conv1 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2=nn.Conv1d(128, 3, kernel_size=1, bias=True)

        self.conv3 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv4=nn.Conv1d(128, 1, kernel_size=1, bias=True)

    def forward(self, pc1, pc2, feature1, feature2):
        l1_pc1_fps, l1_pc1, l1_feature1, fps_index_l1_pc1 = self.sa1(pc1, feature1)
        l2_pc1_fps, l2_pc1, l2_feature1, grouped_index_l2_pc1 = self.sa2(l1_pc1, l1_feature1)

        l1_pc2_fps, l1_pc2, l1_feature2, fps_index_l1_pc2 = self.sa1(pc2, feature2)
        l2_pc2_fps, l2_pc2, l2_feature2, _ = self.sa2(l1_pc2, l1_feature2)

        _, l2_feature1_new = self.fe_layer(l2_pc1, None, l2_pc2, l2_feature1, l2_feature2, 128)

        _, l3_pc1, l3_feature1, fps_index_l3_pc1 = self.sa3(l2_pc1, l2_feature1_new)
        _, l4_pc1, l4_feature1, fps_index_l3_pc1 = self.sa4(l3_pc1, l3_feature1)

        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)

        if self.bn:
            x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        else:
            x = F.relu(self.conv1(l0_fnew1))
        sf_1 = self.conv2(x)

        if self.bn:
            y = F.relu(self.bn2(self.conv3(l0_fnew1)))
        else:
            y = F.relu(self.conv3(l0_fnew1))

        mask = self.conv4(y)
        mask = mask.squeeze(1)
        center_flow = self.fusion(l2_pc1.transpose(2, 1).contiguous(), pc1.transpose(2, 1).contiguous(), sf_1)
        l2_pc1_re = center_flow + l2_pc1

        # ............................... Fusion goes here ...................................
        _, l2_feature1_new_re = self.fe_layer(l2_pc1, l2_pc1_re, l2_pc2, l2_feature1, l2_feature2, 128)
        _, l3_pc1_re, l3_feature1_re, fps_index_l3_pc1_re = self.sa3(l2_pc1, l2_feature1_new_re)
        _, l4_pc1_re, l4_feature1_re, fps_index_l3_pc1_re = self.sa4(l3_pc1_re, l3_feature1_re)

        l3_fnew1_re = self.su1(l3_pc1_re, l4_pc1_re, l3_feature1_re, l4_feature1_re)
        l2_fnew1_re = self.su2(l2_pc1, l3_pc1_re, torch.cat([l2_feature1, l2_feature1_new_re], dim=1), l3_fnew1_re)
        l1_fnew1_re = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1_re)
        l0_fnew1_re = self.fp(pc1, l1_pc1, feature1, l1_fnew1_re)

        if self.bn:
            x_re = F.relu(self.bn1(self.conv1(l0_fnew1_re)))
        else:
            x_re = F.relu(self.conv1(l0_fnew1_re))
        sf_2 = self.conv2(x_re)

        if self.bn:
            y_re = F.relu(self.bn2(self.conv3(l0_fnew1_re)))
        else:
            y_re = F.relu(self.conv3(l0_fnew1_re))
        mask_2 = self.conv4(y_re)
        mask_2 = mask_2.squeeze(1)
        return sf_1, sf_2, mask, mask_2, l1_pc1_fps, l1_pc1, l2_pc1_fps, l2_pc1, l1_pc2_fps, l1_pc2, l2_pc2_fps, l2_pc2, l3_pc1, l4_pc1


def scene_flow_EPE_np(pred, labels, mask):
    error = torch.sqrt(torch.sum((pred - labels)**2, 2) + 1e-20)
    gtflow_len = torch.sqrt(torch.sum(labels*labels, 2) + 1e-20) # B,N
    a = (error <= 0.05).float()*mask
    b = (error/gtflow_len <= 0.05).float()*mask
    c = (error <= 0.1).float()*mask
    d = (error/gtflow_len <= 0.1).float()*mask
    e = a + b
    idx1 = e == 2
    e[idx1] = 1
    f = c + d
    idx2 = f == 2
    f[idx2] = 1
    acc1 = torch.sum((e), 1)
    acc2 = torch.sum((f), 1)
    mask_sum = torch.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = torch.mean(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = torch.mean(acc2)

    EPE_total = torch.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = torch.mean(EPE_total)
    return EPE, acc1, acc2


if __name__ == '__main__':
    pass