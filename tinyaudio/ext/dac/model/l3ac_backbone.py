import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def _init_weight(module: nn.Module):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.trunc_normal_(module.weight, std=0.02)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0.0)
    return module


def Conv1d(*args, **kwargs):
    return weight_norm(_init_weight(nn.Conv1d(*args, **kwargs)))


def Linear(*args, **kwargs):
    return weight_norm(_init_weight(nn.Linear(*args, **kwargs)))


@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor):
    return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


class Snake1d(nn.Module):
    def __init__(self, channels: int, *, channels_last: bool = False):
        super().__init__()
        shape = (1, 1, channels) if channels_last else (1, channels, 1)
        self.alpha = nn.Parameter(torch.ones(*shape))

    def forward(self, x: torch.Tensor):
        return snake(x, self.alpha)


class ChannelNorm(nn.Module):
    def __init__(self, n_channels: int, *, channels_last: bool = False, eps: float = 1e-6):
        super().__init__()
        self.n_channels = n_channels
        self.channels_last = channels_last
        self.weight = nn.Parameter(torch.ones(n_channels))
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        if self.channels_last:
            return F.layer_norm(x, (self.n_channels,), self.weight, self.bias, self.eps)
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return x * self.weight.view(*shape) + self.bias.view(*shape)


class GRN(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, n_channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, n_channels))

    def forward(self, x: torch.Tensor):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class Residual(nn.Module):
    def __init__(self, module: nn.Module, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.module = module
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def _drop_path(self, x: torch.Tensor):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep:
            mask.div_(keep_prob)
        return x * mask

    def forward(self, x: torch.Tensor):
        return x + self._drop_path(self.module(x))


def trend_pool(x: torch.Tensor, kernel_size: int):
    if kernel_size <= 1:
        return x
    pool_args = dict(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return F.avg_pool1d(F.max_pool1d(x.abs(), **pool_args), **pool_args)


class TrendPool(nn.Module):
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor):
        return trend_pool(x, self.kernel_size)


class BaseBlock(nn.Module):
    def __init__(
        self,
        target_dim: int,
        conv_kernels=(7, 7, 7, 7),
        pool_kernels=(1, 3, 5, 9),
        dilation_rate: int = 2,
    ):
        super().__init__()
        assert target_dim % len(pool_kernels) == 0
        branch_dim = target_dim // len(pool_kernels)
        blocks = []
        for conv_kernel, pool_kernel in zip(conv_kernels, pool_kernels):
            conv_dilation = pool_kernel // dilation_rate + 1
            conv_padding = (conv_kernel - 1) * conv_dilation // 2
            blocks.append(
                nn.Sequential(
                    TrendPool(pool_kernel),
                    Conv1d(1, branch_dim, kernel_size=conv_kernel, dilation=conv_dilation, padding=conv_padding),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor):
        return torch.cat([block(x) for block in self.blocks], dim=1)


class V3FirstBlock(BaseBlock):
    def __init__(
        self,
        target_dim: int,
        conv_kernels=(7, 7, 7, 7, 7),
        pool_kernels=(1, 5, 11, 21, 45),
        dilation_rate: int = 99,
    ):
        hidden_dim = len(pool_kernels) * 4
        super().__init__(hidden_dim, conv_kernels, pool_kernels, dilation_rate=dilation_rate)
        self.conv_1 = Conv1d(hidden_dim, hidden_dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.conv_2 = Conv1d(hidden_dim * 4 + 1, target_dim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        h = super().forward(x)
        h = self.act(self.conv_1(h))
        return self.conv_2(torch.cat([h, x], dim=1))


def FirstBlock(dim: int):
    return V3FirstBlock(dim)


class EnhanceBlock(BaseBlock):
    def __init__(self, dim: int):
        super().__init__(4, conv_kernels=(7, 7, 7, 7), pool_kernels=(1, 3, 5, 9))
        self.merge_layer = nn.Sequential(
            nn.InstanceNorm1d(4, affine=True),
            nn.Conv1d(4, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        y = super().forward(x[:, :1, :])
        return x + self.merge_layer(y) * x


class ConvUnit(nn.Module):
    def __init__(self, dim: int, snake_act: bool = True, norm: bool = False, dilation: int = 1, kernel_size: int = 7):
        super().__init__()
        total_pad = (kernel_size - 1) * dilation
        self.dw_conv = Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=total_pad // 2,
            groups=dim,
        )
        self.norm = ChannelNorm(dim, channels_last=True) if norm else nn.Identity()
        self.pw_conv1 = Linear(dim, 4 * dim)
        self.act = Snake1d(4 * dim, channels_last=True) if snake_act else nn.GELU()
        self.grn = GRN(4 * dim)
        self.pw_conv2 = Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor):
        x = self.dw_conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw_conv2(x)
        return x.permute(0, 2, 1)


class LegacyUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        total_pad = (7 - 1) * dilation
        self.block = nn.Sequential(
            Snake1d(dim),
            Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=total_pad // 2),
            Snake1d(dim),
            Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


class L3ACEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 32,
        strides=(4, 4, 5, 5),
        depths=(1, 1, 2, 2, 2),
        dims=(32, 64, 128, 256, 512),
        drop_path_rate: float = 0.0,
        use_norm: bool = True,
        use_snake_act: bool = True,
    ):
        super().__init__()
        blocks = [FirstBlock(dims[0])]
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i_d, o_d, stride, depth in zip(dims[:-1], dims[1:], strides, depths):
            stage = nn.Sequential(
                *[
                    Residual(
                        ConvUnit(i_d, snake_act=use_snake_act, norm=use_norm),
                        drop_prob=drop_path_rates[cur + j],
                    )
                    for j in range(depth)
                ]
            )
            down_layer = nn.Sequential(
                Conv1d(i_d, o_d, kernel_size=stride, stride=stride),
                ChannelNorm(o_d) if use_norm else nn.Identity(),
            )
            blocks.extend([stage, down_layer])
            cur += depth
        blocks.extend(
            [
                nn.Sequential(
                    *[
                        Residual(
                            ConvUnit(dims[-1], snake_act=use_snake_act, norm=use_norm),
                            drop_prob=drop_path_rates[cur + j],
                        )
                        for j in range(depths[-1])
                    ]
                ),
                Conv1d(dims[-1], feature_dim, kernel_size=3, padding=1),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        return self.blocks(x)


class LastBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x)


class L3ACDecoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 32,
        strides=(5, 5, 4, 4),
        depths=(2, 2, 2, 1, 1),
        dims=(512, 256, 128, 64, 32),
        drop_path_rate: float = 0.0,
        use_snake_act: bool = True,
        use_norm: bool = True,
        decoder_last_layer: str = "legacy",
    ):
        super().__init__()
        blocks = [Conv1d(feature_dim, dims[0], kernel_size=3, padding=1)]
        drop_path_rates = [x.item() for x in torch.linspace(drop_path_rate, 0, sum(depths))]
        cur = 0
        for i_d, o_d, stride, depth in zip(dims[:-1], dims[1:], strides, depths):
            stage = nn.Sequential(
                *[
                    Residual(
                        ConvUnit(i_d, snake_act=use_snake_act, norm=use_norm),
                        drop_prob=drop_path_rates[cur + j],
                    )
                    for j in range(depth)
                ]
            )
            up_layer = nn.Sequential(
                Conv1d(i_d, o_d, kernel_size=1, stride=1),
                nn.Upsample(scale_factor=stride, mode="linear", align_corners=False),
                ChannelNorm(o_d) if use_norm else nn.Identity(),
            )
            blocks.extend([stage, EnhanceBlock(i_d), up_layer])
            cur += depth

        if decoder_last_layer is None:
            tail = nn.Sequential(
                *[
                    Residual(ConvUnit(dims[-1], snake_act=use_snake_act, norm=use_norm))
                    for _ in range(2)
                ]
            )
        elif decoder_last_layer == "legacy":
            tail = nn.Sequential(
                Residual(LegacyUnit(dims[-1], dilation=1)),
                Residual(LegacyUnit(dims[-1], dilation=3)),
                Residual(LegacyUnit(dims[-1], dilation=9)),
            )
        elif decoder_last_layer == "dilation":
            tail = nn.Sequential(
                Residual(ConvUnit(dims[-1], snake_act=use_snake_act, norm=use_norm, dilation=1)),
                Residual(ConvUnit(dims[-1], snake_act=use_snake_act, norm=use_norm, dilation=3)),
                Residual(ConvUnit(dims[-1], snake_act=use_snake_act, norm=use_norm, dilation=9)),
            )
        else:
            raise NotImplementedError(decoder_last_layer)

        last_block = LastBlock(
            nn.Sequential(
                tail,
                Snake1d(dims[-1]),
                Conv1d(dims[-1], 1, kernel_size=7, padding=3),
                nn.Tanh(),
            )
        )
        self.blocks = nn.Sequential(*blocks, last_block)

    def forward(self, x: torch.Tensor):
        return self.blocks(x)
