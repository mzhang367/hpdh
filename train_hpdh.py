import os
import sys
import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from cifarDataset import CIFAR100
import torch.nn as nn
from get_map import mAHP_torch, mnDCG_torch
from datetime import datetime
import time
import torchvision.transforms.transforms as transforms
from loss import ClassWiseLoss
import argparse
from networks import back_bone
import torch.optim as optim
from utils import AverageMeter, Logger, MyCustomNabirds, compute_result, compute_mAP
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

"""
The single level class-wise loss is inherited from https://github.com/mzhang367/dcwh
"""

parser = argparse.ArgumentParser(description='PyTorch Implementation of Semantic Hierarchy Preserving Deep Hashing')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--freq', default=80, type=int, help='freq. of print batch information')
parser.add_argument('--path', nargs='+', help='path to loading/saving models, accept multiple arguments as list')
parser.add_argument('--len', nargs='+', type=int, help='length of hashing codes, accept multiple arguments as list, should be (16, 32, 64, 128)')

parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate mode turned on')
parser.add_argument('--dataset', type=str, default='cifar100', help='which dataset to use: {cifar100, nabirds}')

parser.add_argument('--weight_cubic', type=float, default=10)
parser.add_argument('--weight_vertex', type=float, default=0.1)

parser.add_argument('--bs', type=int, default=128, help='Batch size of each iteration')
parser.add_argument('--network', type=str, default='res50',
                    help='which network for training')


args = parser.parse_args()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True
torch.manual_seed(24)


######################################################################################################
Normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.Resize(240),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    Normalize,
    ])

transform_test = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    Normalize,
])

if args.dataset == "cifar100":

    hier_count = [100, 20]
    sigmas = [1, 0.5]
    trainset = CIFAR100(root='./data_cifar100', train=True, download=False, transform=transform_train, coarse=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    testset = CIFAR100(root='./data_cifar100', train=False, download=False, transform=transform_test, coarse=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)


elif args.dataset == "nabirds":

    hier_count = [555, 495, 228, 22]
    sigmas = [1, 0.67, 0.5, 0.25]
    train_labels = np.load("train_labels_nabirds.npy")
    test_labels = np.load("test_labels_nabirds.npy")
    with open('train_test_split.pickle', 'rb') as f:
            train_paths, test_paths = pickle.load(f)
    trainset = MyCustomNabirds(train_paths, train_labels, transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)

    testset = MyCustomNabirds(test_paths, test_labels, transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)


def adjust_learning_rate(optimizer, epoch):

    """Sets the learning rate to the initial LR decayed by 0.5 every 100 epochs"""

    lr = args.lr * (0.1 ** (epoch // 40))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
    return lr


def centers_update(model, data_loader, hier_num, list_hierar_info, len_bit):
    """
    Update centroids recursively of each level of hierarchy
    """
    U = torch.Tensor(len(data_loader.dataset), len_bit).cuda()
    labels = torch.IntTensor(len(data_loader.dataset), len(hier_num)).cuda()
    centers = [torch.Tensor(hier_num[i], len_bit).cuda() for i in range(len(hier_num))]
    for iter, (data, targets) in enumerate(data_loader):
        data_input, target = data.cuda(), targets.cuda()
        output = model(data_input)
        U[iter * args.bs: iter * args.bs + len(data), :] = output
        labels[iter * args.bs: iter * args.bs + len(data), :] = targets

    for i in range(hier_num[0]):
        inner_index = torch.nonzero(torch.eq(labels[:, 0], i), as_tuple=True)[0]
        # find all the index with the label equal to i
        centers[0][i, :] = U[inner_index, :].mean(dim=0)

    for j in range(1, len(hier_num)):
        for k in range(hier_num[j]):    # only for cifar100 dataset, generally, should be pairs of (keys, values)
            centers[j][k, :] = centers[j-1][list_hierar_info[j-1][k], :].mean(dim=0)   # e.g. centers[2][k, :] = centers[1][labels_rel[1][k, :], :]
            # labels_rel contains classes of level 1 constituting level 2

    return centers


def train(len_bit, save_path):

    print("number of training images: ", len(trainset))
    print("number of testing images: ", len(testset))
    print("number of hierarchies of the %s: " % args.dataset, len(hier_count))
    print("number of fine-level classes: ", hier_count[0], end='\n\n')
    print("Hyper parameter:\nsigmas: ",  *sigmas)
    print("weight_cubic: %.4f\t weight_vertex: %.4f" % (args.weight_cubic, args.weight_vertex))

    if args.dataset == "cifar100":
        if not os.path.exists('labels_rel_cifar100.pickle'):
            label_index = []
            super_sub = [[] for i in range(20)]
            list_hierar_info = []
            for batch_idx, (_, targets) in enumerate(trainloader):
                for j in range(targets.shape[0]):
                    if targets[j, 0] not in label_index:
                        label_index.append(targets[j, 0])
                        super_sub[targets[j, 1]].append(targets[j, 0])
            list_hierar_info.append(super_sub)

            with open('labels_rel_cifar100.pickle', 'wb') as f:
                pickle.dump(list_hierar_info, f, protocol=pickle.DEFAULT_PROTOCOL)
        else:

            with open('labels_rel_cifar100.pickle', 'rb') as f:
                list_hierar_info = pickle.load(f)
    else:

        if not os.path.exists('labels_rel_nabirds.pickle'):
            list_hierar_info = []
            for i in range(1, 4):
                label_index = []
                low_to_high = [[] for k in range(hier_count[i])]
                train_labels_level = trainset.labels[:, i-1:i+1]
                for j in range(train_labels_level.shape[0]):
                    if train_labels_level[j, 0] not in label_index:
                        label_index.append(train_labels_level[j, 0])
                        low_to_high[int(train_labels_level[j, 1])].append(int(train_labels_level[j, 0]))
                list_hierar_info.append(low_to_high)

            with open('labels_rel_nabirds.pickle', 'wb') as f:
                pickle.dump(list_hierar_info, f, protocol=pickle.DEFAULT_PROTOCOL)

        else:

            with open('labels_rel_nabirds.pickle', 'rb') as f:
                list_hierar_info = pickle.load(f)

    criterion_beacon = [ClassWiseLoss(num_classes=hier_count[i], bit_length=len_bit, inv_var=sigmas[i])
                        for i in range(len(hier_count))]

    centers_list = [torch.randn(hier_count[i], len_bit).cuda() for i in range(len(hier_count))]
    EPOCHS = 150
    net = back_bone(args.network, len_bit).create_model()
    net = torch.nn.DataParallel(net).to(device)

    for name, module in net.named_children():
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
    optimizer = optim.SGD([
        {'params': net.module.parameters(), 'weight_decay': 5e-4, 'lr': args.lr},
        # {'params': hier_construct.parameters(), 'weight_decay': 5e-4, 'lr': 5e-4},
    ], momentum=0.9)

    schedule = MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)  # [50, 150]

    best_MAP = 0
    best_epoch = 1
    best_loss = 1e4

    torch.autograd.set_detect_anomaly(True)
    since = time.time()
    ##############################################
    for epoch in range(EPOCHS):
        net.train()
        losses = AverageMeter()
        cubic_reg_loss = AverageMeter()
        vertex_reg_loss = AverageMeter()
        print('==> Epoch: %d' % (epoch + 1))
        # adjust_learning_rate(optimizer, epoch)
        ##############################################
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()  # shape of [batch_size, num_hier]
            optimizer.zero_grad()
            features = net(inputs)
            loss_beacon = [criterion_beacon[i](features, targets[:, i], centers_list[i]) for i in range(len(hier_count))]

            loss_cubic = F.relu(-1.1 - features).sum() + F.relu(features - 1.1).sum()
            loss_l2 = loss_cubic / len(inputs)

            Bbatch = torch.sign(features)

            loss_l3 = (Bbatch - features).pow(2).sum() / len(inputs)
            loss = sum(loss_beacon) + (args.weight_cubic * loss_l2 + args.weight_vertex * loss_l3) / len_bit

            losses.update(loss.item(), len(inputs))
            cubic_reg_loss.update(loss_l2.item(), len(inputs))
            vertex_reg_loss.update(loss_l3.item(), len(inputs))
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % args.freq == 0:
                # print('step[{}/{}], loss:{}.4f, Acc: {:.2f}'.format(batch_idx+1, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))
                print("Batch {}/{}\t Loss {:.4f}"  \
                      .format(batch_idx + 1, len(trainloader), losses.avg))

        print('Epoch %d \t Loss: %.4f (cubic_loss: %.4f | vertex_loss: %.4f)'
              % (epoch + 1, losses.avg, cubic_reg_loss.avg, vertex_reg_loss.avg))

        #####################################################################################################
        schedule.step()

        with torch.no_grad():

            centers_list = centers_update(net, trainloader, hier_count, list_hierar_info, len_bit)

            net.eval()

            if (epoch+1) % 5 == 0:  # 10 epochs
                trainB, train_labels = compute_result(trainloader, net, device)
                testB, test_labels = compute_result(testloader, net, device)
                mAP, _ = compute_mAP(trainB, testB, train_labels[:, 0], test_labels[:, 0], device)
                if mAP > best_MAP:
                    best_MAP = mAP

                print("[epoch: %d]\t[hashing loss: %.3f\t [mAP: %.2f%%]" % (epoch+1, losses.avg, float(mAP)*100.0))

        if losses.avg < best_loss:
            print('Saving..')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(net.state_dict(), './checkpoint/%s' % save_path)
            best_loss = losses.avg

    time_elapsed = time.time() - since
    print("Training Completed in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Model saved as %s" % save_path)


def test(len_bit, load_path, check_AHP=True, check_nDCG=True):

    assert os.path.exists(os.path.join("./checkpoint", load_path)), "model path not found!"
    checkpoint = torch.load("./checkpoint/%s" % load_path)
    net = back_bone(args.network, len_bit).create_model()
    net = torch.nn.DataParallel(net).to(device)
    net.load_state_dict(checkpoint)
    net.eval()
    with torch.no_grad():
        trainB, train_labels = compute_result(trainloader, net, device)
        testB, test_labels = compute_result(testloader, net, device)
        mAP, _ = compute_mAP(trainB, testB, train_labels[:, 0], test_labels[:, 0], device)
        print('[Evaluate Phase] MAP: %.2f%%' % (100. * float(mAP)))
        if check_AHP:
            if args.dataset == "cifar100":
                iHPs = np.load("cifar100_imAHPs.npy")
                iHPs = torch.from_numpy(iHPs).to(device)
                mAHP = mAHP_torch(trainB, train_labels, testB, test_labels, iHPs)

            else:
                iHPs = np.load("nabirds_imAHPs.npy")
                iHPs = torch.from_numpy(iHPs).to(device)
                mAHP = mAHP_torch(trainB, train_labels, testB, test_labels, iHPs, r=250, n_level=4)

            print('[Evaluate Phase] mAHP: %.2f%%' % (100. * float(mAHP)))
        if check_nDCG:
            if args.dataset == "cifar100":

                inDCGs = np.load("cifar100_iDCGs.npy")
            else:
                inDCGs = np.load("nabirds_iDCGs.npy")

            inDCGs = torch.from_numpy(inDCGs).to(device)
            nDCG = mnDCG_torch(trainB, train_labels, testB, test_labels, inDCGs)
            print('[Evaluate Phase] nDCG: %.2f%%' % (100. * float(nDCG)))


if __name__ == '__main__':

    if not os.path.isdir('log'):
        os.mkdir('log')
    save_dir = './log'
    # sys.stdout = Logger(osp.join(save_dir, args.network + '_' + args.clf_type + '_' + datetime.now().strftime('%m%d_%H%M') + '.txt'))
    check_AHP = True
    check_nDCG = True
    if args.evaluate:
        assert len(args.path) == len(args.len), 'model paths must be in line with # code lengths'
        for i, (len_bit, save_path) in enumerate(zip(args.len, args.path)):
            test(len_bit, save_path, check_AHP, check_nDCG)
            print('\n')

    else:

        assert len(args.path) == len(args.len), 'model paths must be in line with # code lengths'
        for i, (len_bit, load_path) in enumerate(zip(args.len, args.path)):
            sys.stdout = Logger(os.path.join(save_dir,
                str(len_bit) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M%S') + '.txt'))
            print("[Configuration] Training on dataset: %s\n  Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n "
            %(args.dataset, len_bit, args.bs, args.lr))
            train(len_bit, load_path)


