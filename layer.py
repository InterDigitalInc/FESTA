import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import pointnet2_utils as pointutils


class flow_fusion(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        pos1_t = xyz
        pos2_t = new_xyz

        B, N, C= xyz.shape
        dists,idx = pointutils.knn(16, pos1_t,pos2_t)
        dists[dists < 1e-10] = 1e-10
        weight = 1.0 / dists
        weight = weight / torch.sum(weight, -1,keepdim = True)   # [B,N,3]
        interpolated_feat = torch.sum(pointutils.grouping_operation(features, idx) * weight.view(B, 1, N, 16), dim = -1) # [B,C,N,3]

        return interpolated_feat


class SpatialAbstraction(nn.Module):
    def __init__(self, args,npoint, radius, nsample, in_channel, mlp, group_all):
        super(SpatialAbstraction, self).__init__()
        self.mlp = mlp
        self.bn = args.bn
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel*2+3   # TODO：
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        if group_all:
            self.queryandgroup = pointutils.GroupAll()
        else:
            self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        device = xyz.device
        B, C, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()

        if self.group_all == False:
            fps_idx = pointutils.furthest_point_sample(xyz_t, self.npoint)  # [B, N]
            new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, C, N]
            if points is not None:
                new_feature = pointutils.gather_operation(points, fps_idx)
        else:
            new_xyz = xyz
        new_points, grouped_xyz, grouped_index = self.queryandgroup(xyz_t, new_xyz.transpose(2, 1).contiguous(), points) # [B, 3+C, N, S]
        if points is not None:
            new_points = torch.cat([new_points, new_feature.unsqueeze(-1).repeat(1, 1, 1, self.nsample)], dim=1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0]
        return new_xyz, new_xyz, new_points, fps_idx


class SpatialAbstraction_Attention(nn.Module):
    def __init__(self, args, npoint, radius, nsample, in_channel, mlp, group_all):
        super(SpatialAbstraction_Attention, self).__init__()
        self.mlp = mlp
        self.bn = args.bn
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_convs_all = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp_bns_all = nn.ModuleList()
        self.softmax = nn.Softmax(dim=2)
        last_channel = in_channel*2+3   # TODO：
        last_channel_all = in_channel+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_convs_all.append(nn.Conv2d(last_channel_all, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            self.mlp_bns_all.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
            last_channel_all = out_channel

        if group_all:
            self.queryandgroup = pointutils.GroupAll()
        else:
            self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N] [B, 3, 2048]
            points: input points data, [B, D, N] [B, D, 2048]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        device = xyz.device
        B, C, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        if self.group_all == False:
            fps_idx = pointutils.furthest_point_sample(xyz_t, self.npoint)  # [B, npoint]
            new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, 3, npoint]
            new_feature = pointutils.gather_operation(points, fps_idx)
        else:
            new_xyz = xyz

        new_points_0, grouped_xyz, grouped_index = self.queryandgroup(xyz_t, new_xyz.transpose(2, 1).contiguous(), points) # [B, 3+C, N, S]
        new_points = torch.cat([new_points_0, new_feature.unsqueeze(-1).repeat(1, 1, 1, self.nsample)], dim=1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            if self.bn:
                new_points =  F.relu(bn(conv(new_points)))
            else:
                new_points =  F.relu(conv(new_points))
        new_points_1 = torch.max(new_points, -1)[0]
        new_points_2 = new_points_1.view(B, self.mlp[-1], self.npoint, 1)
        new_points_3 = new_points_2.repeat(1, 1, 1, self.nsample)

        probability = self.softmax(torch.sum((new_points.mul(new_points_3)), 1))
        probability_1 = probability.view(B, 1, self.npoint, self.nsample)
        probability_2 = probability_1.repeat(1, 3, 1, 1)
        new_xyz_1 = torch.sum((grouped_xyz.mul(probability_2)), 3)
        new_points_4, grouped_xyz_1, grouped_index1= self.queryandgroup(xyz_t, new_xyz_1.transpose(2, 1).contiguous(), points)

        for i, conv in enumerate(self.mlp_convs_all):
            bn = self.mlp_bns_all[i]
            new_points_4 =  F.relu(bn(conv(new_points_4)))
        new_points_4 = torch.max(new_points_4, -1)[0]

        return new_xyz, new_xyz_1, new_points_4, grouped_index


class TemporalAbstraction_Attention(nn.Module):
    def __init__(self, args, radius, nsample, in_channel, mlp, pooling='max', corr_func='concat', knn = True):
        super(TemporalAbstraction_Attention, self).__init__()
        self.bn = args.bn
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.pooling = pooling
        self.corr_func = corr_func
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if corr_func is 'concat':
            last_channel = in_channel*2+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, pos1, pos1_re, pos2, feature1, feature2, ksample):
        """
        Input:
            xyz1: (batch_size, 3, npoint)
            xyz2: (batch_size, 3, npoint)
            feat1: (batch_size, channel, npoint)
            feat2: (batch_size, channel, npoint)
        Output:
            xyz1: (batch_size, 3, npoint)
            feat1_new: (batch_size, mlp[-1], npoint)
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()

        B, N, C = pos1_t.shape
        if self.knn:

            if pos1_re is not None:
                pos1_t_re = pos1_re.permute(0, 2, 1).contiguous()
                _, idx = pointutils.knn(ksample, pos1_t_re, pos2_t)
            else:
                _, idx = pointutils.knn(ksample, pos1_t, pos2_t)
        else:
            idx, cnt = query_ball_point(self.radius, self.nsample, pos2_t, pos1_t)
        
            _, idx_knn = pointutils.knn(self.nsample, pos1_t, pos2_t)
            cnt = cnt.view(B, -1, 1).repeat(1, 1, self.nsample)
            idx = idx_knn[cnt > (self.nsample-1)]

        pos2_grouped = pointutils.grouping_operation(pos2, idx) # [B, 3, N, S]
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)    # [B, 3, N, S]

        feat2_grouped = pointutils.grouping_operation(feature2, idx)    # [B, C, N, S]
        if self.corr_func=='concat':
            feat_diff = torch.cat([feat2_grouped, feature1.view(B, -1, N, 1).repeat(1, 1, 1, ksample)], dim = 1)

        feat1_new = torch.cat([pos_diff, feat_diff], dim = 1)  # [B, 2*C+3,N,S]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            if self.bn:
                feat1_new = F.relu(bn(conv(feat1_new)))
            else:
                feat1_new = F.relu(conv(feat1_new))
        feat1_new = torch.max(feat1_new, -1)[0]  # [B, mlp[-1], npoint]

        return pos1, feat1_new
