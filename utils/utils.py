import torch
import torch.nn as nn


class BatchRenorm(nn.Module):
    """
    Batch Re-Normalization layer

    Parameters:
        - num_features: number of features in the input tensor
        - eps: epsilon value to avoid division by zero
        - momentum: momentum to update running statistics
        - rmax: maximum value of r
        - dmax: maximum value of d
        - affine: whether to apply learnable affine transformation
    """    
    def __init__(self, num_features, eps=1e-5, momentum=0.9, 
                rmax=3.0, dmax=5.0, affine=True, warmup_steps=5000):
        super(BatchRenorm, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.rmax = rmax
        self.dmax = dmax
        self.affine = affine
        self.warmup_steps = warmup_steps

        self.register_buffer(
            'running_mean', torch.zeros(num_features, dtype=torch.float)
            )
        self.register_buffer(
            'running_var', torch.ones(num_features, dtype=torch.float)
            )
        self.register_buffer(
            "steps", torch.tensor(0, dtype=torch.long)
        )

        self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(num_features), requires_grad=False)
        self.r = nn.Parameter(torch.ones(1), requires_grad=True)
        self.d = nn.Parameter(torch.zeros(1), requires_grad=True)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.training:
            # computing batch stats
            # TODO: figure out whether the line below is correct
            # dims = [0] + list(range(2, x.dim())) # if x.dim>2, dims=[0, 2, 3, ...]
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            batch_std = (batch_var + self.eps).sqrt()

            # warmup factor for smooth transition from BN to BR
            t = min(1.0, self.steps.item()/ self.warmup_steps)

            if t < 1.0:
                # warmup with BN
                x_norm = (x - batch_mean) / batch_std
            else:
                # compute correction
                running_std = (self.running_var + self.eps).sqrt()
                

                eff_r_max = 1 + t * (self.rmax - 1)
                eff_d_max = t * self.dmax
                r = (((batch_std / running_std) * eff_r_max)
                    .clamp(1.0 / eff_r_max, eff_r_max))
                d = (((batch_mean - self.running_mean) / running_std * eff_r_max)
                     .clamp(-eff_d_max, eff_d_max))
                x_norm = (x - batch_mean) / batch_std * r + d
            
            # update running stats
            self.running_mean.data =  (1 - self.momentum) * batch_mean + self.momentum * self.running_mean
            self.running_var.data = (1 - self.momentum) * batch_var + self.momentum * self.running_var

            self.steps += 1
        else:
            # inference mode
            running_std = (self.running_var + self.eps).sqrt()
            x_norm = (x - self.running_mean) / running_std

        if self.weight is not None:
            x_norm = x_norm * self.weight + self.bias

        return x_norm
