import torch
import torch.nn as nn


class ClassWiseLoss(nn.Module):

    def __init__(self, num_classes, bit_length, inv_var=1, update_grad=False, use_gpu=True):
        super(ClassWiseLoss, self).__init__()
        self.num_classes = num_classes
        self.bits = bit_length
        self.use_gpu = use_gpu
        self.sigma = inv_var
        self.update = update_grad   # update by intra-class hashing outputs or gradient descent
        if update_grad:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.bits).cuda())

    def forward(self, x, labels, centroids=None):
        """
        Args:
            x: batch_size * feat_dim
            labels: (batch_size, )
        """
        if not self.update:
            self.centers = centroids
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)  # shape of (bs * num_bit)
        numer = torch.exp(-0.5*self.sigma*distmat)
        denumer = numer.sum(dim=1, keepdim=True)
        dist_div = numer/(denumer+1e-8)
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.view(-1, 1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist_log = torch.log(dist_div+1e-8) * mask.float()

        loss = -dist_log.sum() / batch_size

        return loss

