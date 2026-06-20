from typing import Any, Dict, List, Literal, Optional
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, model_validator
import optuna
from models.config import UNetConfig


def log_pydantic_config_kv(cfg: BaseModel, logger, *, prefix: str = "cfg"):
    for k, v in cfg.model_dump().items():
        logger.info("%s.%s = %s", prefix, k, v)


class FloatSpec(BaseModel):
    type: Literal["float"]
    low: float
    high: float
    log: bool = False
    step: Optional[float] = None

    @model_validator(mode="after")
    def _check_bounds(self):
        if not (self.low < self.high):
            raise ValueError(
                f"float spec requires low < high (got {self.low} >= {self.high})"
            )
        if self.log and self.low <= 0:
            raise ValueError("log=true requires low > 0 for float spec")
        return self

    def sample(self, trial: optuna.Trial, name: str) -> float:
        return float(
            trial.suggest_float(
                name,
                float(self.low),
                float(self.high),
                log=bool(self.log),
                step=None if self.step is None else float(self.step),
            )
        )


class IntSpec(BaseModel):
    type: Literal["int"]
    low: int
    high: int
    log: bool = False
    step: int = 1

    @model_validator(mode="after")
    def _check_bounds(self):
        if not (self.low < self.high):
            raise ValueError(
                f"int spec requires low < high (got {self.low} >= {self.high})"
            )
        if self.step <= 0:
            raise ValueError("int spec requires step > 0")
        if self.log and self.low <= 0:
            raise ValueError("log=true requires low > 0 for int spec")
        return self

    def sample(self, trial: optuna.Trial, name: str) -> float:
        return int(
            trial.suggest_int(
                name,
                float(self.low),
                float(self.high),
                log=bool(self.log),
                step=None if self.step is None else float(self.step),
            )
        )


# Can't use tuple/array as not serialisable?
class CategoricalSpec(BaseModel):
    type: Literal["categorical"]
    choices: List[Any]

    @model_validator(mode="after")
    def _check_choices(self):
        if len(self.choices) == 0:
            raise ValueError("categorical spec requires non-empty choices")
        norm = [tuple(x) if isinstance(x, list) else x for x in self.choices]
        seen = set()
        deduped = []
        for x in norm:
            if x not in seen:
                seen.add(x)
                deduped.append(x)

        self.choices = deduped
        return self

    def sample(self, trial: optuna.Trial, name: str) -> float:
        return trial.suggest_categorical(name, self.choices)


SearchSpec = FloatSpec | IntSpec | CategoricalSpec


class UNetHPT(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fixed: dict[str, Any] = Field(default_factory=dict)
    choices: dict[str, SearchSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_fixed(self):
        # Optional: validate known fixed keys/types
        if "activation" in self.fixed:
            if self.fixed["activation"] not in ("relu", "silu", "gelu"):
                raise ValueError("unet.fixed.activation must be one of relu|silu|gelu")
        if "upsample_mode" in self.fixed:
            if self.fixed["upsample_mode"] not in (
                "nearest",
                "bilinear",
                "convtranspose",
            ):
                raise ValueError(
                    "unet.fixed.upsample_mode must be nearest|bilinear|convtranspose"
                )
        return self

    def sample(self, trial: optuna.Trial, *, prefix: str = "unet") -> UNetConfig:
        resolved: Dict[str, Any] = dict(self.fixed)
        for k, spec in self.choices.items():
            resolved[k] = spec.sample(trial, f"{prefix}.{k}")
        return UNetConfig(**resolved)


class OptimHPT(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fixed: dict[str, Any] = Field(default_factory=dict)
    choices: dict[str, SearchSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self):
        if "optimizer_name" in self.fixed and self.fixed["optimizer_name"] not in ("adam", "adamw", "sgd"):
            raise ValueError("optim.fixed.optimizer_name must be adam|adamw|sgd")
        return self

    def sample(self, trial: optuna.Trial, *, prefix: str = "optim") -> dict[str, Any]:
        resolved: Dict[str, Any] = dict(self.fixed)
        for k, spec in self.choices.items():
            resolved[k] = spec.sample(trial, f"{prefix}.{k}")
        return resolved


class OptunaStudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    study_name: str = "optuna_study"
    direction: Literal["minimize", "maximize"] = Field(default="minimize")
    n_trials: int = Field(default=1, ge=1)
    storage: str | Path = "sqlite:///optuna.db"


class DataloaderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_path: str | Path = "data"
    batch_size: int = Field(default=64, ge=1)
    num_workers: int = Field(default=0, ge=0)
    transform: str = "default"
    shuffle: bool = True


class HPTYaml(BaseModel):
    model_config = ConfigDict(extra="forbid")
    opt_study_cfg: OptunaStudyConfig = Field(
        default_factory=OptunaStudyConfig, alias="study"
    )
    dl_cfg: DataloaderConfig = Field(
        default_factory=DataloaderConfig, alias="dataloader"
    )
    unet: UNetHPT = Field(default_factory=UNetHPT)
    optim: OptimHPT = Field(default_factory=OptimHPT)

    def sample(self, trial: optuna.Trial) -> tuple[UNetHPT, dict[str, Any]]:
        return self.unet.sample(trial), self.optim.sample(trial)

    def log_config(self, logger) -> None:
        log_pydantic_config_kv(self.opt_study_cfg, logger, prefix="optuna_study")
        log_pydantic_config_kv(self.dl_cfg, logger, "dl")
        log_pydantic_config_kv(self.unet, logger, "unet")
        log_pydantic_config_kv(self.optim, logger, "optim")
