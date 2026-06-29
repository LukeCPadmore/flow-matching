Note to self: 
To start mlflow & optuna servers, in ssh machine, run the following commands:
```shell
cd ~/ml-storage
docker compose up -d
```
Then on local run the following command to forward the ports:
```shell
ssh -L 5000:localhost:5000 \
  -L 5432:localhost:5432 \
  -L 8080:localhost:8080 \
  luke-padmore@luke-padmore-ml
```

## Training
The training entrypoint uses LightningCLI, so run a config directly with `fit`:
```shell
python train.py fit --config configs/cifar10/cifar10_colouriser.yaml
```

To launch a run in `tmux`, start the MLflow Docker Compose stack, and shut the machine down after it completes:
```shell
./run_and_shutdown.sh --config configs/cifar10/cifar10_colouriser.yaml --shutdown
```

Useful variations:
```shell
./run_and_shutdown.sh --config configs/cifar10/cifar10_colouriser_cfg.yaml
./run_and_shutdown.sh --config configs/cifar10/cifar10_colouriser_lencoder.yaml
./run_and_shutdown.sh --config configs/mnist/mnist_uncond.yaml
```

## Optuna Sweeps
For `optuna-lightning-cli`, use the LEncoder tuning config:
```shell
optuna-lightning tune \
  --training-config configs/cifar10/optuna/cifar10_colouriser_lencoder_training.yaml \
  --optuna-config configs/cifar10/optuna/cifar10_colouriser_lencoder_optuna.yaml
```
