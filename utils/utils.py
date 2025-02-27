import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform


class StableTanhTransform(TanhTransform):
    def __init__(self, cache_size=1):
        super().__init__(cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, StableTanhTransform)

    def _inverse(self, y):
        return self.atanh(y)


class SquashedNormal(TransformedDistribution):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = scale
        base_distribution = Normal(loc, scale)
        super().__init__(base_distribution, StableTanhTransform(), validate_args=False)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


warmup_function = {
    "smooth": lambda c, w: min(1.0, c / w),
    "non_smooth": lambda c, w: min(1.0, c // w),
}


class BatchRenorm(nn.Module):
    """
    BatchRenorm - (arxiv.org/abs/1702.03275)

    Args:
        num_features: number of features in input tensor
        momentum: momentum for running statistics
        warmup: number of batches to warmup the batch renorm
        max_r: maximum value for r
        max_d: maximum value for d
        smoothing: smoothing factor for transition from BN to BR
    """

    def __init__(
        self,
        num_features,
        momentum=0.01,
        warmup=100000,
        max_r=3.0,
        max_d=5.0,
        warmup_type="smooth",
    ):
        super(BatchRenorm, self).__init__()
        self.momentum = momentum
        self.warmup = warmup
        self.max_r = max_r
        self.max_d = max_d
        self.smoothing = warmup_function[warmup_type]
        self.batch_size = 0
        self.num_features = num_features
        self.register_buffer("step", torch.zeros(1))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor):
        if not x.dim() >= 2:  # first dim is batch size, second dim is num_features
            raise ValueError("expected 2D input (got {}D input)".format(x.dim()))

        # prepare dimensions for scaling and shifting parameters
        # suppose we have input of shape (batch_size, num_features, height, width) (e.g. (32, 3, 64, 64))
        view_shape = [1, x.shape[1]] + [1] * (x.dim() - 2)  # [1, 3, 1, 1]

        dims = [i for i in range(x.dim()) if i != 1]  # [0, 2, 3]

        running_std = (self.running_var + self.eps).sqrt()

        if self.training:
            mean = x.mean(dims)
            var = x.var(dims, unbiased=False)
            std = (var + self.eps).sqrt()

            r = torch.clamp(std / running_std, 1 / self.max_r, self.max_r)
            d = torch.clamp(
                (mean - self.running_mean) / running_std, -self.max_d, self.max_d
            )

            if self.step < self.warmup:
                # BatchNorm
                smoothing_factor = self.smoothing(self.step.item(), self.warmup)
                r = 1.0 + (r - 1.0) * smoothing_factor
                d = d * smoothing_factor

            # update running statistics
            x = (x - mean.view(view_shape)) / std.view(view_shape) * r.view(
                view_shape
            ) + d.view(view_shape)

            raw_var = var.detach() * x.shape[0] / (x.shape[0] - 1)
            self.running_mean += self.momentum * (mean.detach() - self.running_mean)
            self.running_var += self.momentum * (raw_var - self.running_var)

            self.step += 1

        else:
            # inference time
            x = (x - self.running_mean.view(view_shape)) / running_std.view(view_shape)

        return x * self.weight.view(view_shape) + self.bias.view(view_shape)
