import torch
import torch.nn as nn
import torchvision.transforms as transforms


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range=1,
        norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(norm_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(norm_mean) / std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for p in self.parameters():
            p.requires_grad = False


class perceptual_loss(nn.Module):

    def __init__(self, vgg):
        super(perceptual_loss, self).__init__()
        self.normalization_mean = [0.485, 0.456, 0.406]
        self.normalization_std = [0.229, 0.224, 0.225]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = MeanShift(norm_mean=self.normalization_mean, norm_std=self.normalization_std).to(self.device)
        self.vgg = vgg
        self.criterion = nn.MSELoss()

    def forward(self, HR, SR, layer='relu5_4'):
        ## HR and SR should be normalized [0,1]
        hr = self.transform(HR)
        sr = self.transform(SR)

        hr_feat = getattr(self.vgg(hr), layer)
        sr_feat = getattr(self.vgg(sr), layer)

        return self.criterion(hr_feat, sr_feat), hr_feat, sr_feat


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# Now, for the VGG model import and instantiation, assuming you're using torchvision's pre-trained VGG19
import torchvision.models as models

# Assuming you want to use VGG19 as the perceptual loss network
vgg19 = models.vgg19(pretrained=True).features.eval()

# Finally, let's create an instance of the perceptual loss module
perceptual_loss_module = perceptual_loss(vgg19)
