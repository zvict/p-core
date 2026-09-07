import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from math import exp
from .lpips import LPIPS


class BasicLoss(nn.Module):
    def __init__(self, losses_and_weights):
        super(BasicLoss, self).__init__()
        self.losses_and_weights = losses_and_weights

    def forward(self, pred, target):
        loss = 0
        for name_and_weight, loss_func in self.losses_and_weights.items():
            name, weight = name_and_weight.split('/')
            if float(weight) > 0:
                cur_loss = loss_func(pred, target)
                loss += float(weight) * cur_loss
            # print(name, weight, cur_loss, loss)
        return loss
    

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):
        return 1 - ssim(pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2))


class ReverseHuberLoss(nn.Module):
    def __init__(self, delta=0.0005):
        super(ReverseHuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target):
        abs_diff = torch.abs(pred - target)
        # Shift L2 loss so that at abs_diff == self.delta, L1 == L2
        # L1 at boundary = self.delta
        # Unscaled L2 at boundary = 0.5 * self.delta^2
        # Shift needed: self.delta - 0.5 * self.delta^2
        l2_shift = self.delta - 0.5 * self.delta ** 2
        shifted_l2 = 0.5 * (pred - target) ** 2 + l2_shift
        return torch.mean(torch.where(abs_diff < self.delta, abs_diff, shifted_l2))


class AdaptiveReverseHuberLoss(nn.Module):
    def __init__(self):
        super(AdaptiveReverseHuberLoss, self).__init__()

    def forward(self, pred, target):
        abs_diff = torch.abs(pred - target)
        C = abs_diff.max().item() * 0.2
        scaled_l2 = ((pred - target) ** 2 + C*C)/(2*C)
        return torch.mean(torch.where(abs_diff < C, abs_diff, scaled_l2))    


# The SSIM implementation below is adapted from
# https://github.com/Po-Hsun-Su/pytorch-ssim, MIT licensed.
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
    

def ssim(img1, img2, window_size=11, size_average=True):
    if img2.device != img1.device or img2.dtype != img1.dtype:
        img2 = img2.to(device=img1.device, dtype=img1.dtype)

    channel = img1.size(-3)
    window = create_window(window_size, channel)
    window = window.to(device=img1.device, dtype=img1.dtype)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-6)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def get_loss(args, bias=1.0):
    losses = nn.ModuleDict()
    for loss_name, weight in args.items():
        if weight > 0:
            if loss_name == "mse":
                losses[loss_name + "/" +
                       str(format(weight, '.0e'))] = nn.MSELoss()
                print("Using MSE loss, loss weight: ", weight)
            elif loss_name == "l1":
                losses[loss_name + "/" +
                       str(format(weight, '.0e'))] = nn.L1Loss()
                print("Using L1 loss, loss weight: ", weight)
            elif loss_name == "lpips":
                lpips = LPIPS(net='vgg')
                losses[loss_name + "/" + str(format(weight, '.0e'))] = lpips
                print("Using LPIPS loss, loss weight: ", weight)
            elif loss_name == "lpips_alex":
                lpips = LPIPS(net='alex')
                losses[loss_name + "/" + str(format(weight, '.0e'))] = lpips
                print("Using LPIPS AlexNet loss, loss weight: ", weight)
            elif loss_name == "ssim":
                losses[loss_name + "/" + str(format(weight, '.0e'))] = SSIMLoss()
                print("Using SSIM loss, loss weight: ", weight)
            elif loss_name == "berhu":
                losses[loss_name + "/" + str(format(weight, '.0e'))] = ReverseHuberLoss()
                print("Using Reverse Huber loss, loss weight: ", weight)
            elif loss_name == "ada_berhu":
                losses[loss_name + "/" + str(format(weight, '.0e'))] = AdaptiveReverseHuberLoss()
                print("Using Adaptive Reverse Huber loss, loss weight: ", weight)
            else:
                raise NotImplementedError(
                    'loss [{:s}] is not supported'.format(loss_name))
    return BasicLoss(losses)

