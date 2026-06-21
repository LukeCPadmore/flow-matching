from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.cli import LightningCLI


def main() -> None:
    LightningCLI(
        model_class=LightningModule,
        datamodule_class=LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=False,
    )


if __name__ == "__main__":
    main()
