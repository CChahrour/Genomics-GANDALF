import argparse
import datetime
import json
import os
import sys

import pandas as pd
import pytorch_tabular
import torch
import wandb
from loguru import logger
from pytorch_tabular import TabularModel
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.models import GANDALFConfig


def setup_logger(results_dir: str):
    logger.remove()

    log_dir = os.path.join(results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "run.log")

    logger.add(
        sink=sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        colorize=True,
    )
    logger.add(
        sink=log_file,
        level="INFO",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        colorize=False,
        enqueue=True,
    )


@logger.catch
def load_data(data_dir: str, region_name: str, target: str, target_column: str):
    data = pd.read_parquet(f"{data_dir}/{region_name}.parquet")

    for col in data.select_dtypes(include=["float64"]).columns:
        data[col] = data[col].astype("float32")

    meth_cols = [col for col in data.columns if "METH" in col]
    data[meth_cols] = data[meth_cols].fillna(-1)

    X_data = data[[col for col in data.columns if "SEM" in col and target not in col]]
    y_data = data[target_column]

    dataset = pd.concat([X_data, y_data], axis=1)

    train_data = dataset[~dataset.index.str.startswith(("chr8", "chr9"))]
    val_data = dataset[dataset.index.str.startswith("chr8")]
    test_data = dataset[dataset.index.str.startswith("chr9")]

    return train_data, val_data, test_data


@logger.catch
def train(train_data, val_data, test_data, project, group, results_dir, device, target):
    with wandb.init(
        name=f"run_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}",
        project=project,
        group=group,
        job_type="sweep",
        dir=os.path.join(results_dir, "wandb"),
        reinit="finish_previous",
    ) as run:
        config = run.config

        data_config = DataConfig(
            continuous_cols=[col for col in train_data.columns if target not in col],
            dataloader_kwargs={"persistent_workers": True},
            normalize_continuous_features=False,
            num_workers=8,
            pin_memory=True,
            target=[col for col in train_data.columns if target in col],
            validation_split=0,
        )

        optimizer_config = OptimizerConfig()

        trainer_config = TrainerConfig(
            accelerator="mps" if device.type == "mps" else "gpu",
            auto_lr_find=False,
            batch_size=config.batch_size,
            check_val_every_n_epoch=5,
            checkpoints_path=os.path.join(results_dir, "checkpoints"),
            early_stopping_mode="min",
            early_stopping_patience=3,
            early_stopping="valid_loss",
            load_best=True,
            max_epochs=config.max_epochs,
            progress_bar="rich",
            trainer_kwargs=dict(enable_model_summary=False),
        )

        experiment_config = ExperimentConfig(
            exp_log_freq=5,
            exp_watch="gradients",
            log_logits=False,
            log_target="wandb",
            project_name=project,
            run_name=run.name,
        )

        model_config = GANDALFConfig(
            embedding_dropout=config.embedding_dropout,
            gflu_dropout=config.gflu_dropout,
            gflu_feature_init_sparsity=config.gflu_feature_init_sparsity,
            gflu_stages=config.gflu_stages,
            learning_rate=config.lr,
            head="LinearHead",
            loss="MSELoss",
            metrics=["r2_score", "mean_squared_error"],
            metrics_params=[{}] * 2,
            seed=42,
            target_range=[(0, 1)],
            task="regression",
        )

        model = TabularModel(
            data_config=data_config,
            experiment_config=experiment_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=False,
            suppress_lightning_logger=True,
        )

        model.fit(train=train_data, validation=val_data)
        model.predict(test_data)


def main():
    parser = argparse.ArgumentParser(description="Train a GANDALF model.")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Dataset directory."
    )
    parser.add_argument("--target", type=str, default="MLL-N", help="Target column.")
    parser.add_argument(
        "--region_name", type=str, default="promoters_1024bp", help="Region name."
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="SEM_CAT_1_MLL-N",
        help="Target column name in the dataset.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="SEM_MLL-N_TF",
        help="WandB project name.",
    )
    
    args = parser.parse_args()

    model = "GANDALF_SEM"
    project = args.project
    task = "singlelabel_regression"
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    group = f"{model}_{args.region_name}_{args.target}_{task}"
    results_dir = f"results/{project}/{group}_{start_time}"
    os.makedirs(results_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = results_dir

    setup_logger(results_dir)

    logger.info(f"Project: {project} | Group: {group}")

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(
            f"Using GPU: {torch.cuda.get_device_name(0)} | Count: {torch.cuda.device_count()}"
        )

    logger.info(
        f"Versions: Torch: {torch.__version__}, PyTorch Tabular: {pytorch_tabular.__version__}"
    )

    train_data, val_data, test_data = load_data(
        args.data_dir, args.region_name, args.target, args.target_column
    )

    logger.info("🧹 Sweep")
    with open("config/sweep_config.json", "r") as f:
        sweep_config = json.load(f)

    sweep_config["name"] = group
    try:
        sweep_id = wandb.sweep(sweep_config, project=project)
    except wandb.errors.CommError:
        logger.warning("Sweep already exists or WANDB issue, reusing sweep ID")
        sweep_id = sweep_config.get("sweep_id")

    wandb.agent(
        sweep_id=sweep_id,
        function=lambda: train(
            train_data,
            val_data,
            test_data,
            project,
            group,
            results_dir,
            device,
            args.target,
        ),
        count=50,
        project=project,
    )

    logger.success("🏁 Sweep finished")

    api = wandb.Api()
    logger.info(sweep_id)
    sweep = api.sweep(f"catherine-chahrour-university-of-oxford/{project}/{sweep_id}")
    best_run = sorted(
        sweep.runs, key=lambda r: r.summary.get("valid_r2_score", 0), reverse=True
    )[0]
    logger.success(
        f"Best run: {best_run.id} | R²: {best_run.summary['valid_r2_score']}"
    )

    config = best_run.config
    with open(os.path.join(results_dir, "best_run_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    logger.info(f"Results saved in: {results_dir}")
    logger.info("👋 Shutting down EC2 instance...")
    os.system("sudo shutdown -h now")

if __name__ == "__main__":
    main()
