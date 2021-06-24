import torch
import numpy as np
from torch.utils.data import DataLoader
from cifarDataset import CIFAR100
import torchvision.transforms as transforms
from utils import *


def compute_imAHP(trainset, testset, hier_list, radius=2500):

    '''
    pre-compute ideal mAHP
    '''

    train_labels = trainset.labels

    test_labels = testset.labels

    unique_query_label = list(test_labels[:, 0])

    imAHPs = np.ndarray((hier_list[0], radius))
    for i in range(hier_list[0]):
        query_index = unique_query_label.index(i)  # return the first index
        query_label = test_labels[query_index, :]   # construct a tuple of (subclass, superclass)
        comp = np.array(np.equal(train_labels, query_label), dtype=np.float)
        imatch = np.sort(np.sum(comp, axis=1), kind='mergesort', axis=0)[::-1]  # in descending order
        imatch = imatch[:radius]/len(hier_list)   # [0, 0.5, 1]
        imAHPs[i, :] = np.cumsum(imatch)
    return imAHPs


def compute_iDCG(trainset, testset, hierar_list, base=2, radius=100):

    '''
    pre-compute ideal DCG
    '''

    train_labels = trainset.labels
    test_labels = testset.labels

    unique_fine_label = list(test_labels[:, 0])

    iDCGs = np.ndarray((hierar_list[0], radius))
    for i in range(hierar_list[0]):
        query_index = unique_fine_label.index(i)
        query_label = test_labels[query_index, :]

        comp = np.array(np.equal(train_labels, query_label), dtype=np.int)
        imatch = np.sort(np.sum(comp, axis=1), kind='mergesort')[::-1]   # Gain vector
        iDCG = np.ndarray(radius)
        iDCG[:base] = np.cumsum(imatch)[:base]
        for j in range(base, radius):
            iDCG[j] = iDCG[j-1] + imatch[j]/np.log2(j)
        iDCGs[i, :] = iDCG
    return iDCGs


def mAHP_topk(train_codes, train_labels, test_codes, test_labels, iHPs, top_list, n_level):
    """
    calculate mAHP at given positions in top_list
    """
    mAHP_array = np.ndarray(len(top_list))
    for j, rank in enumerate(top_list):
        AHPx = []
        for i in range(test_codes.size(0)):
            query_label, query_binary = test_labels[i, :], test_codes[i]
            _, query_result = torch.sum((query_binary != train_codes).int(), dim=1).sort()  # broadcast, return tuple of (values, indices)
            correct = torch.sum((query_label == train_labels[query_result[:rank], :]).float(), dim=1)  # here, .int() cause 0.5 --> 0 !!!
            rele_num = torch.sum(correct)
            correct/=n_level
            ls = torch.cumsum(correct.squeeze(), dim=0) # from cumsum to sum: become HP?
            iHP = iHPs[query_label[0], :rank]   # index from look-up table,
            Px = ls.float() / iHP.float()  # here, we need to construct iHP in advance
            if rele_num != 0:
                AHPx.append(torch.mean(Px))     # note, it is mean(Px)
            else:
                AHPx.append(0)
        AHP = torch.tensor(AHPx)
        mAHP = torch.mean(AHP)
        mAHP_array[j] = mAHP

    return mAHP_array


def mAHP_torch(train_codes, train_labels, test_codes, test_labels, iHPs, r=2500, n_level=2):
    """
    Use pre-computed mAHPs
    """

    AHPx = []

    for i in range(test_codes.size(0)):
        query_label, query_binary = test_labels[i, :], test_codes[i]
        _, query_result = torch.sum((query_binary != train_codes).int(), dim=1).sort()  # broadcast, return tuple of (values, indices)
        correct = torch.sum((query_label == train_labels[query_result[:r], :]).float(), dim=1)
        rele_num = torch.sum(correct)
        correct/=n_level
        ls = torch.cumsum(correct.squeeze(), dim=0)
        iHP = iHPs[query_label[0], :]
        Px = ls.float() / iHP.float()
        if rele_num != 0:
            AHPx.append(torch.mean(Px))
        else:
            AHPx.append(0)
    AHP = torch.tensor(AHPx)
    mAHP = torch.mean(AHP)
    return mAHP


def mnDCG_torch(train_codes, train_labels, test_codes, test_labels, iDCGs, base=2, r=100):

    """
    Use pre-computed iDCGs
    Note that label indices in fine labels must correspond with that in iDCGs
    """

    nDCGs = []

    for i in range(test_codes.size(0)):
        query_label, query_binary = test_labels[i, :], test_codes[i]
        _, query_result = torch.sum((query_binary != train_codes).int(), dim=1).sort()
        correct = torch.sum((query_label == train_labels[query_result[:r], :]).int(), dim=1).squeeze()
        rele_num = torch.sum(correct)
        DCG = torch.Tensor(r).cuda()
        DCG[:base] = torch.cumsum(correct, dim=0)[:base]
        for j in range(base, r):
            DCG[j] = DCG[j-1] + correct[j]/torch.log2(torch.tensor(j).float())

        iDCG = iDCGs[query_label[0]]
        nDCG = DCG / iDCG.float()
        if rele_num != 0:
            nDCGs.append(torch.mean(nDCG))
        else:
            nDCGs.append(0)

    A_nDCG = torch.mean(torch.tensor(nDCGs))
    return A_nDCG


if __name__ == '__main__':

    dataset = "nabirds"

    if dataset == "cifar100":
        trainset = CIFAR100(root='./data_cifar100', train=True, download=False, transform=None, coarse=True)
        testset = CIFAR100(root='./data_cifar100', train=False, download=False, transform=None, coarse=True)
        hier_list = [100, 20]
    elif dataset == "nabirds":
        with open('train_test_split_list.pickle', 'rb') as f:
            train_paths, test_paths = pickle.load(f)
        train_labels = np.load("train_labels_birds.npy")
        test_labels = np.load("test_labels_birds.npy")

        trainset = MyCustomNabirds(train_paths, train_labels)
        testset = MyCustomNabirds(test_paths, test_labels)
        hier_list = [555, 495, 228, 22]

    imAHPs = compute_imAHP(trainset, testset, hier_list, radius=250)    # r@2.5k for cifar100
    np.save("%s_imAHPs.npy" % dataset, imAHPs)
    iDCG = compute_iDCG(trainset, testset, hier_list)
    np.save("%s_iDCGs.npy" % dataset, iDCG)

