import torch
import torch.nn as nn
import math
from src.flow_matching.models.config import UNetConfig


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        trunk_dim: int = 32,
        max_period: float = 10000.0,
        activation_cls: type[nn.Module] | None = None,
    ):
        super().__init__()
        assert embedding_dim % 2 == 0, "Use an even embedding dimension."
        self.embedding_dim = embedding_dim
        self.log_max_period = math.log(max_period)

        act_cls = activation_cls if activation_cls is not None else nn.SiLU

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, trunk_dim),
            act_cls(),
            nn.Linear(trunk_dim, trunk_dim),
        )

    def _sinusoidal_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        t = t.squeeze()
        B = t.shape[0]
        half = self.embedding_dim // 2

        i = torch.arange(half, device=t.device, dtype=torch.float32)
        inv_freq = torch.exp(-self.log_max_period * (i / half))  # (half,)
        angles = t[:, None] * inv_freq[None, :]  # (B, half)

        emb = torch.stack([angles.sin(), angles.cos()], dim=-1)  # (B, half, 2)
        return emb.flatten(start_dim=1)  # (B, embedding_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal_time_embedding(t))


class SimpleClassConditioning(nn.Module):
    def __init__(
        self,
        n_classes,
        embedding_dim,
        trunk_dim,
        activation_cls: type[nn.Module] | None = None,
    ):
        super().__init__()
        self.cond_emb = nn.Embedding(n_classes, embedding_dim)
        act_cls = activation_cls if activation_cls else nn.SiLU
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, trunk_dim),
            act_cls(),
            nn.Linear(trunk_dim, trunk_dim),
        )

    def forward(self, cls_idx):
        cls_embedding = self.cond_emb(cls_idx)
        return self.mlp(cls_embedding)


class GreyScaleEncoder(nn.Module):
    def __init__(
        self,
        d_trunk: int,
        hidden_channels: int | None = None,
        activation_cls: type[nn.Module] | None = None,
    ):
        super().__init__()
        hidden_channels = int(hidden_channels or max(8, d_trunk))
        act_cls = activation_cls if activation_cls is not None else nn.SiLU
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            act_cls(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            act_cls(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, d_trunk),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.encoder(x)


def conv_gn_act(
    in_ch: int,
    out_ch: int,
    *,
    groups: int,
    act_cls: nn.Module,
    kernel_size=3,
    stride: int = 1,
    bias: bool = False,
    padding: int = 1,
    p_drop: float = 0.0,
) -> list[nn.Module]:
    """
    Pre-activation block: GN -> Act -> Conv -> (optional Dropout2d)
    """
    layers: list[nn.Module] = [
        nn.GroupNorm(groups, in_ch),
        act_cls(),
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        ),
    ]
    if p_drop > 0:
        layers.append(nn.Dropout2d(p_drop))
    return layers


def two_conv_block(
    in_ch: int,
    mid_ch: int,
    out_ch: int,
    *,
    groups: int,
    act_cls: nn.Module,
    p_drop: float = 0.0,
) -> nn.Sequential:
    """
    Two 3x3 convs with pre-activation and optional dropout after each conv.
    """
    layers: list[nn.Module] = []
    layers += conv_gn_act(
        in_ch, mid_ch, groups=groups, act_cls=act_cls, stride=1, p_drop=p_drop
    )
    layers += conv_gn_act(
        mid_ch, out_ch, groups=groups, act_cls=act_cls, stride=1, p_drop=p_drop
    )
    return nn.Sequential(*layers)


class ConvDownblock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        d_trunk,
        d_concat,
        group_norm_size=8,
        activation_cls: type[nn.Module] | None = None,
        p_drop=0,
    ):
        """
        Deciding to use stride in order to downsample instead of maxpooling
        """
        super().__init__()
        self.d_concat = d_concat
        self.group_norm_size = group_norm_size
        act_cls = activation_cls if activation_cls is not None else nn.SiLU

        self.conv = two_conv_block(
            in_ch=in_channels + d_concat,
            mid_ch=out_channels,
            out_ch=out_channels,
            groups=group_norm_size,
            act_cls=act_cls,
            p_drop=p_drop,
        )

        # downsample conv (stride=2)
        self.down = nn.Sequential(
            *conv_gn_act(
                in_ch=out_channels,
                out_ch=out_channels,
                groups=group_norm_size,
                act_cls=act_cls,
                stride=2,
                p_drop=p_drop,
            )
        )
        self.emb_mlp = nn.Linear(d_trunk, d_concat)

    def forward(self, x: torch.Tensor, trunk_emb: torch.Tensor) -> torch.Tensor:
        """
        Project trunk embedding to d_concat, expand spatially, then concat.
        """
        emb = self.emb_mlp(trunk_emb)[:, :, None, None]
        emb = emb.expand(-1, -1, x.size(-2), x.size(-1))
        x = torch.cat([x, emb], dim=1)
        x_skip_features = self.conv(x)
        x = self.down(x_skip_features)
        return x, x_skip_features


class ConvUpblock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        d_trunk,
        d_concat,
        group_norm_size=8,
        upsample_mode="nearest",
        activation_cls: type[nn.Module] | None = None,
        p_drop=0,
    ):
        """
        Note input will be sized (B, 2 * in_channels + d_concat, H, W)
        Channels:
        1 x input channels for residual features
        1 x input channels for x
        1 x d_concat for broadcast trunk embeddings
        """
        super().__init__()
        self.d_concat = d_concat
        self.group_norm_size = group_norm_size
        self.upsampler = UNetConfig.make_upsample(upsample_mode, in_channels)
        act_cls = activation_cls if activation_cls else nn.SiLU
        self.conv = two_conv_block(
            in_ch=2 * in_channels + d_concat,
            mid_ch=out_channels,
            out_ch=out_channels,
            groups=group_norm_size,
            act_cls=act_cls,
            p_drop=p_drop,
        )
        self.emb_mlp = nn.Linear(d_trunk, d_concat)

    def forward(
        self,
        x: torch.Tensor,
        trunk_emb: torch.Tensor,
        skip_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project trunk embedding to d_concat, expand spatially, then concat.
        """
        x = self.upsampler(x)
        emb = self.emb_mlp(trunk_emb)[:, :, None, None]
        emb = emb.expand(-1, -1, x.size(-2), x.size(-1))
        x = torch.cat([x, skip_features, emb], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        channels: list[int],
        d_trunk,
        d_concat,
        group_norm_size=8,
        activation_cls: type[nn.Module] | None = None,
        dropout_enc_list=None,
    ):
        """
        For each channel create DownConvblock with in_channels = channel[i] and out_channels = channels[i+1]
        """
        super().__init__()
        # Lift number of channel to make GroupNorm work
        self.channels = channels.copy()
        self.initial_conv = nn.Conv2d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            padding=1,
            stride=1,
        )
        dropout_enc_list = dropout_enc_list or [0] * (len(self.channels) - 1)
        self.channels[0] = self.channels[1]
        self.down_blocks = nn.ModuleList(
            [
                ConvDownblock(
                    in_channels,
                    out_channels,
                    d_trunk,
                    d_concat,
                    group_norm_size=group_norm_size,
                    activation_cls=activation_cls,
                    p_drop=p_drop,
                )
                for in_channels, out_channels, p_drop in zip(
                    self.channels[:-1], self.channels[1:], dropout_enc_list
                )
            ]
        )

    def forward(self, x, trunk_emb):
        """
        Loop through downblocks and save output tensors to skip_features to later pass to decoder
        """
        skip_features = []
        x = self.initial_conv(x)
        for block in self.down_blocks:
            x, x_skip_features = block(x, trunk_emb)
            skip_features.append(x_skip_features)
        return x, skip_features


class Decoder(nn.Module):
    def __init__(
        self,
        channels: list[int],
        d_trunk,
        d_concat,
        group_norm_size=8,
        upsample_mode="nearest",
        activation_cls: type[nn.Module] | None = None,
        dropout_dec_list=None,
        out_channels: int | None = None,
    ):
        """
        Expects list of channels with the same orders as te encoder
        """
        super().__init__()
        chls = channels.copy()
        in_channels = chls.pop(0)
        self.out_channels = int(in_channels if out_channels is None else out_channels)
        self.channels = chls[::-1]
        self.channels.append(self.channels[-1])
        dropout_dec_list = dropout_dec_list or [0] * (len(self.channels) - 1)
        self.up_blocks = nn.ModuleList(
            [
                ConvUpblock(
                    in_channels,
                    out_channels,
                    d_trunk,
                    d_concat,
                    group_norm_size=group_norm_size,
                    upsample_mode=upsample_mode,
                    activation_cls=activation_cls,
                    p_drop=p_drop,
                )
                for in_channels, out_channels, p_drop in zip(
                    self.channels[:-1], self.channels[1:], dropout_dec_list
                )
            ]
        )
        self.final_conv = nn.Conv2d(
            self.channels[-1], self.out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(
        self, x, trunk_emb: torch.Tensor, skip_features: list[torch.Tensor]
    ) -> torch.Tensor:
        for block in self.up_blocks:
            x = block(x, trunk_emb, skip_features.pop())
        return self.final_conv(x)


class Bottleneck(nn.Module):
    def __init__(
        self,
        channels,
        d_trunk,
        d_concat,
        group_norm_size=8,
        activation_cls: type[nn.Module] | None = None,
        p_drop=0,
    ):
        """
        Uses similar architecture to ConvDownblock but doesn't use stride = 2 to halve image size
        """
        super().__init__()
        self.channels = channels
        self.d_trunk = d_trunk
        self.d_concat = d_concat
        act_cls = activation_cls if activation_cls else nn.SiLU

        self.bottleneck = two_conv_block(
            in_ch=channels + d_concat,
            mid_ch=channels,
            out_ch=channels,
            groups=group_norm_size,
            act_cls=act_cls,
            p_drop=p_drop,
        )
        self.emb_mlp = nn.Linear(d_trunk, d_concat, bias=False)

    def forward(self, x: torch.Tensor, trunk_emb: torch.Tensor) -> torch.Tensor:
        """
        Project trunk embedding to d_concat, expand spatially, then concat.
        """
        emb = self.emb_mlp(trunk_emb)[:, :, None, None]
        emb = emb.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, emb], dim=1)
        return self.bottleneck(x)


class UNet(nn.Module):
    def __init__(
        self,
        channels,
        out_channels: int | None = None,
        d_trunk=32,
        d_concat=8,
        group_norm_size=8,
        d_time=128,
        max_time_period=10000.0,
        activation_cls: type[nn.Module] | None = None,
        upsample_mode="nearest",
        dropout_enc_dec_list=[],
        dropout_bottleneck=0,
        conditioning_model: nn.Module | None = None,
    ):
        super().__init__()
        self.channels = list(channels)
        self.d_trunk = d_trunk
        self.conditioning_model = conditioning_model
        self.encoder = Encoder(
            channels,
            d_trunk,
            d_concat,
            group_norm_size,
            activation_cls=activation_cls,
            dropout_enc_list=dropout_enc_dec_list,
        )
        self.decoder = Decoder(
            channels,
            d_trunk,
            d_concat,
            group_norm_size,
            upsample_mode=upsample_mode,
            activation_cls=activation_cls,
            dropout_dec_list=dropout_enc_dec_list[::-1],
            out_channels=out_channels,
        )
        self.bottleneck = Bottleneck(
            self.channels[-1],
            d_trunk,
            d_concat,
            group_norm_size,
            activation_cls=activation_cls,
            p_drop=dropout_bottleneck,
        )

        self.time_embedding_mlp = SinusoidalTimeEmbedding(
            embedding_dim=d_time,
            trunk_dim=d_trunk,
            max_period=max_time_period,
            activation_cls=activation_cls,
        )
        act_cls = activation_cls if activation_cls is not None else nn.SiLU
        self.cond_time_mlp = nn.Sequential(
            nn.Linear(2 * d_trunk, d_trunk),
            act_cls(),
            nn.Linear(d_trunk, d_trunk),
        )

    def forward(self, x, t, cond_emb=None, *args, **kwargs) -> torch.Tensor:
        time_emb = self.time_embedding_mlp(t)
        if cond_emb is None:
            cond_emb = torch.zeros_like(time_emb)
        elif self.conditioning_model is not None and cond_emb.dim() != 2:
            cond_emb = self.conditioning_model(cond_emb)

        trunk_emb = self.cond_time_mlp(torch.cat([time_emb, cond_emb], dim=1))
        x, skip_features = self.encoder(x, trunk_emb)
        x = self.bottleneck(x, trunk_emb)
        x = self.decoder(x, trunk_emb, skip_features)
        return x

    @classmethod
    def from_config(
        cls,
        cfg: UNetConfig,
        conditioning_model: nn.Module | None = None,
    ) -> "UNet":
        return cls(
            channels=list(cfg.channels),
            out_channels=cfg.out_channels,
            d_trunk=cfg.d_trunk,
            d_concat=cfg.d_concat,
            group_norm_size=cfg.group_norm_size,
            d_time=cfg.d_time,
            max_time_period=cfg.max_time_period,
            activation_cls=cfg.activation_cls,
            upsample_mode=cfg.upsample_mode,
            dropout_enc_dec_list=cfg.dropout_enc_dec_list,
            dropout_bottleneck=cfg.dropout_bottleneck,
            conditioning_model=conditioning_model,
        )


class UncondUNet(nn.Module):
    def __init__(self, core: UNet):
        super().__init__()
        self.core = core

    def forward(self, x, t):
        cond_emb = torch.zeros(
            x.size(0), self.core.d_trunk, device=x.device, dtype=x.dtype
        )
        return self.core(x, t, cond_emb)


class ClassCondUNet(nn.Module):
    def __init__(self, core: UNet, n_classes: int, d_cls_emb: int = 128):
        super().__init__()
        self.core = core
        self.class_cond = SimpleClassConditioning(n_classes, d_cls_emb, core.d_trunk)

    def forward(self, x, t, y):
        cond_emb = self.class_cond(y)  # (B, d_trunk)
        return self.core(x, t, cond_emb)


class SimpleColouriser(nn.Module):
    def __init__(self, core: UNet):
        super().__init__()
        self.core = core

    @classmethod
    def from_config(cls, cfg: UNetConfig) -> "SimpleColouriser":
        core = UNet.from_config(cfg)
        return cls(core)

    def forward(self, x, t):
        return self.core(x, t)
