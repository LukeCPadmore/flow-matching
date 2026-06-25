from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping

import torch.nn as nn

if TYPE_CHECKING:
    pass


OptimizerName = Literal["adam", "adamw", "sgd"]
ActivationName = Literal["relu", "silu", "gelu"]
UpsampleMode = Literal["nearest", "bilinear", "convtranspose"]


def _mlflow_value(v: Any, *, seq_sep: str = ",") -> Any:
    """Convert a value into something MLflow log_params can handle nicely."""
    if v is None:
        return "null"

    if isinstance(v, (list, tuple)):
        return seq_sep.join(map(str, v))

    if isinstance(v, Mapping):
        # You can choose to raise instead if you want strictness
        return str(dict(v))

    return v


def dataclass_to_mlflow_params(
    obj: Any, *, prefix: str = "", seq_sep: str = ","
) -> dict[str, Any]:
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass instance, got {type(obj)}")

    out: dict[str, Any] = {}
    for f in fields(obj):
        k = f.name
        v = getattr(obj, k)
        out[k] = _mlflow_value(v, seq_sep=seq_sep)

    if prefix:
        return {f"{prefix}.{k}": v for k, v in out.items()}
    return out


@dataclass(frozen=True)
class UNetConfig:
    # architecture
    in_channels: int = 1
    out_channels: int | None = None
    base_channels: int = 16
    mult: float = 2  # e.g. 16 -> 32 in next layer with
    n_layers: int = 3
    d_trunk: int = 32
    d_concat: int = 8
    group_norm_size: int = 8

    # Linear increaseing dropout in enconder, max dropout in bottleneck and linearly decreasing dropout in decoder
    dropout_enc_dec_min: float = 0
    dropout_enc_dec_max: float = 0
    dropout_bottleneck: float = 0

    # time embedding params
    d_time: int = 128
    max_time_period: float = 10000.0

    # Upsampling and activation function
    activation_name: ActivationName = "silu"
    upsample_mode: UpsampleMode = "nearest"

    @classmethod
    def make_dropout(
        cls, dropout_min: float, dropout_max: float, n_layers
    ) -> list[float]:
        return [
            dropout_min + i * ((dropout_max - dropout_min) / n_layers)
            for i in range(n_layers)
        ]

    @classmethod
    def make_activation(cls, name: ActivationName) -> type[nn.Module]:
        name = name.lower()
        if name == "relu":
            return nn.ReLU
        if name == "silu":
            return nn.SiLU
        if name == "gelu":
            return nn.GELU
        raise ValueError(f"Unknown activation {name}")

    @classmethod
    def make_upsample(cls, name: UpsampleMode, channels: int):
        if name == "nearest":
            return nn.Upsample(scale_factor=2, mode="nearest")
        if name == "bilinear":
            return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        if name == "convtranspose":
            return nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        raise ValueError(f"Unknown upsample {name}")

    @classmethod
    def make_channels(
        cls, in_ch: int, base: int, mult: float, num_layers: int
    ) -> tuple[int, ...]:
        chans = [in_ch]
        c = base
        for _ in range(num_layers):
            chans.append(int(round(c)))
            c *= mult
        return tuple(chans)

    @property
    def channels(self) -> tuple[int, ...]:
        return self.make_channels(
            in_ch=int(self.in_channels),
            base=int(self.base_channels),
            mult=float(self.mult),
            num_layers=int(self.n_layers),
        )

    @property
    def dropout_enc_dec_list(self) -> list[float]:
        return UNetConfig.make_dropout(
            self.dropout_enc_dec_min, self.dropout_enc_dec_max, self.n_layers
        )

    @property
    def activation_cls(self) -> nn.Module:
        return UNetConfig.make_activation(self.activation_name)

    @classmethod
    def _check_groupnorm(cls, C: int, g: int, where: str) -> None:
        if g <= 0:
            raise ValueError(f"{where}: group_norm_size must be > 0")
        if C <= 0:
            raise ValueError(f"{where}: num_channels must be > 0")
        if g > C:
            raise ValueError(f"{where}: group_norm_size ({g}) > num_channels ({C})")
        if C % g != 0:
            raise ValueError(f"{where}: {C} % {g} != 0")

    @classmethod
    def _check_dropout(
        cls, dropout_enc_dec_min, dropout_enc_dec_max, dropout_bottleneck
    ) -> None:
        if dropout_enc_dec_min < 0 or dropout_enc_dec_min >= 1.0:
            raise ValueError(
                f"Encoder-Decoder min dropout must be 0 <= p <= 1, curently set to p = {dropout_enc_dec_min}"
            )
        if dropout_enc_dec_max < 0 or dropout_enc_dec_max >= 1.0:
            raise ValueError(
                f"Encoder-Decoder max dropout must be 0 <= p <= 1, curently set to p = {dropout_enc_dec_min}"
            )
        if dropout_bottleneck < 0 or dropout_bottleneck >= 1.0:
            raise ValueError(
                f"Bottleneck dropout must be 0 <= p <= 1, curently set to p = {dropout_enc_dec_min}"
            )
        if dropout_enc_dec_min > dropout_enc_dec_max:
            raise ValueError(
                f"Encoder-Decoder min dropout (currently {dropout_enc_dec_min}) must smaller than max dropout (currently {dropout_enc_dec_max})"
            )

    def __post_init__(self):
        g = self.group_norm_size
        chans = self.channels

        for level in range(1, len(chans)):
            in_ch = chans[level]
            out_ch = chans[level]  # or whatever you set for that block

            UNetConfig._check_groupnorm(
                in_ch + self.d_concat, g, f"downblock preconv level {level}"
            )
            UNetConfig._check_groupnorm(
                2 * in_ch + self.d_concat, g, f"upblock preconv level {level}"
            )
            UNetConfig._check_groupnorm(out_ch, g, f"postconv level {level}")

        UNetConfig._check_dropout(
            self.dropout_enc_dec_min, self.dropout_enc_dec_max, self.dropout_bottleneck
        )

    def to_mlflow_params(self, *, prefix: str = "unet") -> dict[str, Any]:
        return dataclass_to_mlflow_params(self, prefix=prefix)


def make_optimizer(
    params,
    *,
    optimizer_name: OptimizerName = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        import torch.optim

        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        import torch.optim

        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )
    if optimizer_name == "sgd":
        import torch.optim

        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimiser {optimizer_name}")


def log_config_kv(cfg: Any, logger, *, prefix: str = "unet"):
    if is_dataclass(cfg):
        items = asdict(cfg).items()
    elif isinstance(cfg, Mapping):
        items = cfg.items()
    else:
        raise TypeError(f"Expected dataclass or mapping, got {type(cfg)}")

    for k, v in items:
        logger.info("%s.%s = %s", prefix, k, v)
