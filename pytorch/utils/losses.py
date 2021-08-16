import torch
import torch.nn as nn
from torch.nn.functional import mse_loss as mse
import math

def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    r"""Creates a function that calculates the PSNR between 2 images.

    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    Given an m x n image, the PSNR is:

    .. math::

        \text{PSNR} = 10 \log_{10} \bigg(\frac{\text{MAX}_I^2}{MSE(I,T)}\bigg)

    where

    .. math::

        \text{MSE}(I,T) = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1} [I(i,j) - T(i,j)]^2

    and :math:`\text{MAX}_I` is the maximum possible input value
    (e.g for floating point images :math:`\text{MAX}_I=1`).

    Args:
        input: the input image with arbitrary shape :math:`(*)`.
        labels: the labels image with arbitrary shape :math:`(*)`.
        max_val: The maximum value in the input tensor.

    Return:
        the computed loss as a scalar.

    Examples:
        >>> ones = torch.ones(1)
        >>> psnr(ones, 1.2 * ones, 2.) # 10 * log(4/((1.2-1)**2)) / log(10)
        tensor(20.0000)

    Reference:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(target)}.")

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(input)}.")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")
    loss = torch.log(max_val ** 2 / mse(input, target, reduction='mean'))
    return -1.0 * torch.abs(loss)

def psnr_loss(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    r"""Function that computes the PSNR loss.

    The loss is computed as follows:

     .. math::

        \text{loss} = -\text{psnr(x, y)}

    See :meth:`~kornia.losses.psnr` for details abut PSNR.

    Args:
        input: the input image with shape :math:`(*)`.
        labels : the labels image with shape :math:`(*)`.
        max_val: The maximum value in the input tensor.

    Return:
        the computed loss as a scalar.

    Examples:
        >>> ones = torch.ones(1)
        >>> psnr_loss(ones, 1.2 * ones, 2.) # 10 * log(4/((1.2-1)**2)) / log(10)
        tensor(-20.0000)
    """

    return 1.0 * psnr(input, target, max_val)

class PSNRLoss(nn.Module):
    r"""Creates a criterion that calculates the PSNR loss.

    The loss is computed as follows:

     .. math::

        \text{loss} = -\text{psnr(x, y)}

    See :meth:`~kornia.losses.psnr` for details abut PSNR.

    Args:
        max_val: The maximum value in the input tensor.

    Shape:
        - Input: arbitrary dimensional tensor :math:`(*)`.
        - Target: arbitrary dimensional tensor :math:`(*)` same shape as input.
        - Output: a scalar.

    Examples:
        >>> ones = torch.ones(1)
        >>> criterion = PSNRLoss(2.)
        >>> criterion(ones, 1.2 * ones) # 10 * log(4/((1.2-1)**2)) / log(10)
        tensor(-20.0000)
    """

    def __init__(self, max_val: float) -> None:
        super(PSNRLoss, self).__init__()
        self.max_val: float = max_val

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return psnr_loss(input, target, self.max_val)


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / len(loss1) + len(loss2)

class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
