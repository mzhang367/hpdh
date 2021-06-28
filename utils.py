# coding: utf-8
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import sys
import errno
import os.path as osp


class MyCustomNabirds(Dataset):
    def __init__(self, imagePaths, np_array_label, transform=None):
        # stuff
        self.imagePaths = imagePaths
        self.transforms = transform
        self.labels = np_array_label

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):

        imagePaths = self.imagePaths
        img = Image.open(imagePaths[index]).convert('RGB')
        label = self.labels[index, :]   # a vector of (4, )
        label = torch.from_numpy(label)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label.long()


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise



'''def CalcHammingDist(B1, B2):#test* train
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))# the same
    return distH'''


def CalcHammingDist(B1, B2):#test* train
    q = B2.shape[1]
    distH = 0.5 * (q - torch.mm(B1, B2.transpose(0, 1)))# the same
    return distH


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()# first reset

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def compute_mAP(trn_binary, tst_binary, trn_label, tst_label, device, top=None):

    '''
    mAP @ all
    '''

    AP = []
    top_p = []
    top_mAP = 0
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()  # broadcast, return tuple of (values, indices)
        correct = (query_label == trn_label[query_result]).float()
        N = torch.sum(correct)
        Ns = torch.arange(1, N+1).float().to(device)
        index = (correct.nonzero(as_tuple=False) + 1)[:, 0:1].squeeze(dim=1).float()
        AP.append(torch.mean(Ns / index))
        if top is not None:
            top_result = query_result[:top]
            top_correct = (query_label == trn_label[top_result]).float()
            N_top = torch.sum(top_correct)
            top_p.append(1.0*N_top/top)

    top_mAP = torch.mean(torch.Tensor(top_p))

    mAP = torch.mean(torch.Tensor(AP))
    return mAP, top_mAP


def compute_topK(trn_binary, tst_binary, trn_label, tst_label, device, top_list):

    top_p = torch.Tensor(tst_binary.size(0), len(top_list)).to(device)

    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()#broadcast, return tuple of (values, indices)
        for j, top in enumerate(top_list):
            top_result = query_result[:top]
            top_correct = (query_label == trn_label[top_result]).float()
            N_top = torch.sum(top_correct)
            top_p[i, j] = 1.0*N_top/top

    top_pres = top_p.mean(dim=0).cpu().numpy()

    return top_pres


def compute_result(dataloader, net, device, centers=None):

    hash_codes = []
    label = []
    for i, (imgs, cls, *_) in enumerate(dataloader):
        imgs, cls = imgs.to(device), cls.to(device)
        hash_values = net(imgs)
        if centers is not None:
            center_distance = 0.5*(hash_values.size(1) - torch.mm(torch.sign(hash_values.data), centers.t()))
            hash_code = centers[torch.argsort(center_distance, dim=1)[:, 0]]
            hash_codes.append(hash_code)
        else:
            hash_codes.append(hash_values.data)

        label.append(cls)

    B = torch.sign(torch.cat(hash_codes))

    return B, torch.cat(label)


def EncodingOnehot(target, nclasses):
    target_onehot = torch.Tensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.cpu().view(-1, 1), 1)#
    return target_onehot


#if __name__ == '__main__':

    # [FaceScrub] train, test , classes: 67177, 2650, 530 # [28, 208] for train; 5/class for test
    #  [YouTube] train, test , classes: 63800, 7975, 1595 #40/class for train, 5/class for test
