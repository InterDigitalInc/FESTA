from __future__ import print_function

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import SceneflowDataset, SceneflowDataset_kitti
from tqdm import tqdm
from easydict import EasyDict
from festa import FESTA, FESTA_Kitti, scene_flow_EPE_np


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


def test_one_epoch(args, net, test_loader, textio):
    net.eval()
    total_loss = 0
    num_examples = 0
    epe_3d_sum = 0
    acc_3d_sum = 0
    acc_3d_2_sum = 0
    epe_3d_sum1 = 0
    acc_3d_sum1 = 0
    acc_3d_2_sum1 = 0
    mask_sum = 0
    mask_sum_2 = 0

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        pc1, pc2, color1, color2, flow, mask1 = data
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        color1 = color1.cuda().transpose(2,1).contiguous()
        color2 = color2.cuda().transpose(2,1).contiguous()
        flow = flow.cuda()
        mask1 = mask1.cuda().float()

        batch_size = pc1.size(0)
        num_examples += batch_size
        if args.rgb:
            with torch.no_grad():
                flow_pred, flow_pred_2, mask_pred, mask_pred_2, l1_pc1_fps, l1_pc1, l2_pc1_fps, l2_pc1, l1_pc2_fps, l1_pc2, l2_pc2_fps, l2_pc2, l3_pc1, l4_pc1 = net(pc1, pc2, color1, color2)
        else:
            with torch.no_grad():
                flow_pred, flow_pred_2, mask_pred, mask_pred_2, l1_pc1_fps, l1_pc1, l2_pc1_fps, l2_pc1, l1_pc2_fps, l1_pc2, l2_pc2_fps, l2_pc2, l3_pc1, l4_pc1 = net(pc1, pc2, None, None)

        f_flow = flow_pred.permute(0,2,1)
        if args.recurrent:
            f_flow_2 = flow_pred_2.permute(0,2,1)

        loss = torch.mean(mask1 * torch.sum((f_flow - flow) * (f_flow - flow), -1) / 2.0)
        total_loss += loss.item() * batch_size
        epe_3d, acc_3d, acc_3d_2 = scene_flow_EPE_np(f_flow, flow, mask1)
        if args.recurrent:
            epe_3d1, acc_3d1, acc_3d_21 = scene_flow_EPE_np(f_flow_2, flow, mask1)

        criterion_mask = nn.BCEWithLogitsLoss()
        if mask_pred is not None:
            loss_mask = criterion_mask(mask_pred, mask1)
        else:
            loss_mask = None
        if mask_pred_2 is not None:
            loss_mask_2 = criterion_mask(mask_pred_2, mask1)
        else:
            loss_mask_2 = None
        
        epe_3d_sum += epe_3d.item() * batch_size
        acc_3d_sum += acc_3d.item() * batch_size
        acc_3d_2_sum += acc_3d_2.item() * batch_size
        if args.mask:
            mask_sum += loss_mask.item() * batch_size

        if args.recurrent:
            epe_3d_sum1 += epe_3d1.item() * batch_size
            acc_3d_sum1 += acc_3d1.item() * batch_size
            acc_3d_2_sum1 += acc_3d_21.item() * batch_size
            if args.mask:
                mask_sum_2 += loss_mask_2.item() * batch_size

    return total_loss * 1.0 / num_examples, epe_3d_sum * 1.0 / num_examples, acc_3d_sum * 1.0 / num_examples, acc_3d_2_sum * 1.0 / num_examples, mask_sum * 1.0 / num_examples,\
            epe_3d_sum1 * 1.0 / num_examples, acc_3d_sum1 * 1.0 / num_examples, acc_3d_2_sum1 * 1.0 / num_examples, mask_sum_2 * 1.0 / num_examples


def train_one_epoch(args, net, train_loader, opt, epoch, writer):
    net.train()
    num_examples = 0
    total_loss = 0
    epe_3d_sum = 0
    acc_3d_sum = 0
    acc_3d_2_sum = 0
    epe_3d_sum1 = 0
    acc_3d_sum1 = 0
    acc_3d_2_sum1 = 0
    mask_sum = 0
    mask_sum_2 = 0

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        pc1, pc2, color1, color2, flow, mask1 = data
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        color1 = color1.cuda().transpose(2,1).contiguous()
        color2 = color2.cuda().transpose(2,1).contiguous()
        flow = flow.cuda().transpose(2,1).contiguous()
        mask1 = mask1.cuda().float()

        batch_size = pc1.size(0)
        opt.zero_grad()
        num_examples += batch_size
        if args.rgb:
            flow_pred, flow_pred_2, mask_pred, mask_pred_2, l1_pc1_fps, l1_pc1, l2_pc1_fps, l2_pc1, l1_pc2_fps, l1_pc2, l2_pc2_fps, l2_pc2, l3_pc1, l4_pc1 = net(pc1, pc2, color1, color2)
        else:
            flow_pred, flow_pred_2, mask_pred, mask_pred_2, l1_pc1_fps, l1_pc1, l2_pc1_fps, l2_pc1, l1_pc2_fps, l1_pc2, l2_pc2_fps, l2_pc2, l3_pc1, l4_pc1 = net(pc1, pc2, None, None)

        f_flow = flow_pred.transpose(2,1).contiguous()
        if args.recurrent:
            f_flow_2 = flow_pred_2.transpose(2,1).contiguous()

        criterion_mask = nn.BCEWithLogitsLoss()
        if mask_pred is not None:
            loss_mask = criterion_mask(mask_pred, mask1)
        else:
            loss_mask = None
        if mask_pred_2 is not None:
            loss_mask_2 = criterion_mask(mask_pred_2, mask1)
        else:
            loss_mask_2 = None

        epe_3d, acc_3d, acc_3d_2 = scene_flow_EPE_np(f_flow, flow.transpose(2,1).contiguous(), mask1)
        if args.recurrent:
            if args.one_loss:
                epe_3d1, acc_3d1, acc_3d_21 = scene_flow_EPE_np(f_flow + f_flow_2, flow.transpose(2,1).contiguous(), mask1)
            else:
                epe_3d1, acc_3d1, acc_3d_21 = scene_flow_EPE_np(f_flow_2, flow.transpose(2,1).contiguous(), mask1)

        if args.recurrent:
            if args.mask:
                if args.one_loss:
                    loss_sum = epe_3d1 + 0.3*loss_mask_2
                else:
                    loss_sum = 0.3*epe_3d + epe_3d1 + 0.2*loss_mask + 0.3*loss_mask_2
            else:
                if args.one_loss:
                    loss_sum = epe_3d1
                else:
                    loss_sum = 0.3*epe_3d + epe_3d1
        else:
            if args.mask:
                loss_sum = 0.7*epe_3d + 0.3*loss_mask
            else:
                loss_sum = epe_3d

        loss_sum.backward()
        opt.step()
        total_loss += loss_sum.item() * batch_size
        epe_3d_sum += epe_3d.item() * batch_size
        acc_3d_sum += acc_3d.item() * batch_size
        acc_3d_2_sum += acc_3d_2.item() * batch_size
        if args.recurrent:
            epe_3d_sum1 += epe_3d1.item() * batch_size
            acc_3d_sum1 += acc_3d1.item() * batch_size
            acc_3d_2_sum1 += acc_3d_21.item() * batch_size
        if args.mask:
            mask_sum += loss_mask.item() * batch_size
            if args.recurrent:
                mask_sum_2 += loss_mask_2.item() * batch_size
    return total_loss * 1.0 / num_examples, epe_3d_sum * 1.0 / num_examples, epe_3d_sum1 * 1.0 / num_examples, mask_sum * 1.0 / num_examples, mask_sum_2 * 1.0 / num_examples


def test(args, net, test_loader, boardio, textio):

    test_loss, test_epe_3d, test_acc_3d, test_3d_2_sum, test_mask_sum, test_epe_3d1, test_acc_3d1, test_3d_2_sum1, test_mask_sum_2 = test_one_epoch(args, net, test_loader, textio)
    textio.cprint('==FINAL TEST==')
    textio.cprint('mean test loss: %f'%test_loss)
    textio.cprint('mean test epe_3d: %f'%test_epe_3d)
    textio.cprint('mean test acc_3d: %f'%test_acc_3d)
    textio.cprint('mean test 3d_2_sum: %f'%test_3d_2_sum)
    textio.cprint('mean test mask_sum: %f'%test_mask_sum)
    textio.cprint('mean test epe_3d end: %f'%test_epe_3d1)
    textio.cprint('mean test acc_3d end: %f'%test_acc_3d1)
    textio.cprint('mean test 3d_2_sum end: %f'%test_3d_2_sum1)
    textio.cprint('mean test mask_sum end: %f'%test_mask_sum_2)


def exp_lr_scheduler(optimizer, global_step, init_lr, decay_steps, decay_rate, lr_clip, staircase=True):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if staircase:
        lr = init_lr * decay_rate**(global_step // decay_steps)
    else:
        lr = init_lr * decay_rate**(global_step / decay_steps)
    lr = max(lr, lr_clip)

    if global_step % decay_steps == 0:
        print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(args, net, train_loader, test_loader, boardio, textio):
    if not args.pretrain and not args.resume:
        if args.use_sgd:
            print("Use SGD")
            opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
        else:
            print("Use Adam")
            opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    writer = SummaryWriter('checkpoints/' + args.exp_name + '/log')
    best_test_loss = np.inf

    if args.pretrain:
        model_path = 'checkpoints' + '/' + 'test' + '/models/%s.best.t7'%args.pretrain_name
        textio.cprint("loading checkpoint from: %s"%model_path)
        checkpoint = torch.load(model_path)
        pretrained_dict = checkpoint
        if torch.cuda.device_count() > 1:
            init_dict = net.module.state_dict()
        else:
            init_dict = net.state_dict()
        pretrained_dict_new = {}
        for k, v in init_dict.items():
            if k in pretrained_dict:

                para = pretrained_dict[k]
            else:
                para = v
            pretrained_dict_new[k] = para

        init_dict.update(pretrained_dict_new)
        if torch.cuda.device_count() > 1:
            net.module.load_state_dict(pretrained_dict_new)
        else:
            net.load_state_dict(pretrained_dict_new)
        if args.use_sgd:
            print("Use SGD")
            opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
        else:
            print("Use Adam")
            opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
        textio.cprint("checkpoint loaded successfully")

    if args.resume:
        if args.use_sgd:
            print("Use SGD")
            opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
        else:
            print("Use Adam")
            opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
        model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
        textio.cprint("loading checkpoint from: %s"%model_path)
        checkpoint = torch.load(model_path)
        if torch.cuda.device_count() > 1:
            net.module.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint['state_dict'])
        opt.load_state_dict(checkpoint['optimizer'])
        textio.cprint("checkpoint loaded successfully")

    for epoch in range(args.epochs):
        global_step = epoch * len(train_loader) * args.batch_size
        lr = exp_lr_scheduler(opt, global_step, args.lr, args.decay_steps, args.decay_rate, 0.00001, staircase=True)
        textio.cprint("checking///////")
        textio.cprint(str(global_step))
        textio.cprint(str(lr))
        textio.cprint('==epoch: %d=='%epoch)
        train_loss, train_epe3d_middle, train_epe3d_end, train_mask_middle, train_mask_end = train_one_epoch(args, net, train_loader, opt, epoch, writer)
        textio.cprint('mean train loss: %f'%train_loss)
        textio.cprint('mean train EPE_middle loss: %f'%train_epe3d_middle)
        textio.cprint('mean train EPE_end loss: %f'%train_epe3d_end)
        textio.cprint('mean train mask_middle loss: %f'%train_mask_middle)
        textio.cprint('mean train mask_end loss: %f'%train_mask_end)
        writer.add_scalar('mean train loss',
                        train_loss,
                        epoch * len(train_loader))
        writer.add_scalar('mean train EPE_middle loss',
                        train_epe3d_middle,
                        global_step)
        writer.add_scalar('mean train EPE_end loss',
                        train_epe3d_end,
                        global_step)
        writer.add_scalar('mean train mask_middle loss',
                        train_mask_middle,
                        global_step)
        writer.add_scalar('mean train mask_end loss',
                        train_mask_end,
                        global_step)

        test_loss, test_epe_3d, test_acc_3d, test_3d_2_sum, test_mask_sum, test_epe_3d1, test_acc_3d1, test_3d_2_sum1, test_mask_sum_2 = test_one_epoch(args, net, test_loader, textio)
        textio.cprint('mean test loss: %f'%test_loss)
        textio.cprint('mean test epe_3d: %f'%test_epe_3d)
        textio.cprint('mean test acc_3d: %f'%test_acc_3d)
        textio.cprint('mean test 3d_2_sum: %f'%test_3d_2_sum)
        textio.cprint('mean test mask_sum: %f'%test_mask_sum)
        textio.cprint('mean test epe_3d end: %f'%test_epe_3d1)
        textio.cprint('mean test acc_3d end: %f'%test_acc_3d1)
        textio.cprint('mean test 3d_2_sum end: %f'%test_3d_2_sum1)
        textio.cprint('mean test mask_sum end: %f'%test_mask_sum_2)

        if args.recurrent:
            best_epe = test_epe_3d1
        else:
            best_epe = test_epe_3d

        if best_test_loss >= best_epe:
            best_test_loss = best_epe
            textio.cprint('best test epe loss till now: %f'%best_epe)
            checkpoint_dir = 'checkpoints/%s/models/model.best.t7' % args.exp_name
            if torch.cuda.device_count() > 1:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': net.module.state_dict(),
                    'optimizer': opt.state_dict()
                }
                torch.save(checkpoint, checkpoint_dir)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)


def parse_args_from_yaml(yaml_path):
    with open(yaml_path, 'r') as fd:
        args = yaml.safe_load(fd)
        args = EasyDict(d=args)
    return args


def main():

    args = parse_args_from_yaml(sys.argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    # CUDA settings
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = []
    if not args.eval:
        _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    if args.dataset == 'flythings3d':
        train_loader = DataLoader(
            SceneflowDataset(npoints=args.num_points, root = args.dataset_path, partition='train'),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            SceneflowDataset(npoints=args.num_points, root = args.dataset_path, partition='test'),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'kitti':
        train_loader = DataLoader(
            SceneflowDataset_kitti(npoints=args.num_points, root = args.dataset_path, train=True),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            SceneflowDataset_kitti(npoints=args.num_points, root = args.dataset_path, train=False),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("not implemented")

    if args.model == 'FESTA':
        net = FESTA(args).cuda()
        net.apply(weights_init)
        if args.eval:
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
                print(model_path)
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['state_dict'])
            print(checkpoint['epoch'])
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")

    elif args.model == 'kitti':
        net = FESTA_Kitti(args, args.num_structure_points).cuda()
        net.apply(weights_init)
        if args.eval:
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['state_dict'])
            print(checkpoint['epoch'])
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')
    if args.eval:
        print("starting testing.........................")
        print(len(test_loader))
        test(args, net, test_loader, boardio, textio)
    else:
        print("starting training.........................")
        print(len(train_loader))
        train(args, net, train_loader, test_loader, boardio, textio)
    print('FINISH')

if __name__ == '__main__':
    main()
