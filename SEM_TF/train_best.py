import argparse
import datetime
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
from sklearn.metrics import mean_squared_error, r2_score


@logger.catch
def setup_logger(results_dir: str):
    """Setup logger for console and file output."""
    logger.remove()

    log_file = os.path.join(results_dir, "run.log")

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
def load_data(data_dir: str, region_name: str, target: str):
    data = pd.read_parquet(f"{data_dir}/{region_name}.parquet")

    for col in data.select_dtypes(include=["float64"]).columns:
        data[col] = data[col].astype("float32")

    meth_cols = [col for col in data.columns if "METH" in col]
    data[meth_cols] = data[meth_cols].fillna(-1)

    X_data = data[[col for col in data.columns if "SEM" in col and target not in col]]
    y_data = data["SEM_CAT_1_MLL-N"]

    dataset = pd.concat([X_data, y_data], axis=1)

    train_data = dataset[~dataset.index.str.startswith(("chr8", "chr9"))]
    val_data = dataset[dataset.index.str.startswith("chr8")]
    test_data = dataset[dataset.index.str.startswith("chr9")]

    return train_data, val_data, test_data


@logger.catch
def build_model(train_data, config, results_dir, device, project, target):
    """Configure and instantiate the TabularModel."""
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
        batch_size=config["batch_size"],
        check_val_every_n_epoch=5,
        checkpoints_path=os.path.join(results_dir, "checkpoints"),
        early_stopping_mode="min",
        early_stopping_patience=3,
        early_stopping="valid_loss",
        load_best=True,
        max_epochs=config["max_epochs"],
        progress_bar="rich",
        trainer_kwargs=dict(enable_model_summary=False),
    )

    experiment_config = ExperimentConfig(
        exp_log_freq=5,
        exp_watch=None,
        log_logits=False,
        log_target="wandb",
        project_name=project,
        run_name=f"full_run_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M')}",
    )

    model_config = GANDALFConfig(
        embedding_dropout=config["embedding_dropout"],
        gflu_dropout=config["gflu_dropout"],
        gflu_feature_init_sparsity=config["gflu_feature_init_sparsity"],
        gflu_stages=config["gflu_stages"],
        learning_rate=config["lr"],
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

    return model


@logger.catch
def main(args, results_dir):
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    task = "singlelabel_regression"
    group = f"{args.model_name}_{args.region_name}_{args.target}_{task}"
    results_dir = f"results/{args.project}/full_{group}_{start_time}"

    os.environ["WANDB_DIR"] = results_dir
    os.makedirs(results_dir, exist_ok=True)
    setup_logger(results_dir)

    logger.info(f"Project: {args.project} | Group: {group}")

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    logger.info(f"Loading best config from {args.best_config_path}")
    with open(args.best_config_path, "r") as f:
        config = json.load(f)

    config_path = os.path.join(results_dir, "best_run_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Best config saved to {config_path}")

    logger.info("Loading dataset...")
    train_data, val_data, test_data = load_data(
        args.data_dir, args.region_name, args.target
    )

    logger.info("Building model...")
    model = build_model(
        train_data, config, results_dir, device, args.project, args.target
    )

    wandb.init(
        project=args.project,
        group=group,
        name=f"full_run_{start_time}",
        dir=results_dir,
    )

    logger.info("Training final model...")
    model.fit(train=train_data, validation=val_data)

    logger.info("Preparing model for inference saving...")
    model.model.cpu()
    model.callbacks = None
    model_path = os.path.join(results_dir, "final_model")
    # model.save_model(model_path, inference_only=True)
    model.save_model(model_path, inference_only=False)
    logger.success(f"✅ Final model trained and saved to {model_path}")


    logger.info("Predicting on test data...")
    preds = model.predict(test_data)

    preds_path = os.path.join(results_dir, "test_predictions.parquet")
    preds.to_parquet(preds_path, index=False)
    logger.success(f"✅ Test predictions saved to {preds_path}")

    # Calculate R2 and MSE
    # Get true labels and predictions
    y_true = test_data[
        [col for col in test_data.columns if args.target in col]
    ].values.flatten()
    pred_col = preds.columns[0]
    y_pred = preds[pred_col].values.flatten()


    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    logger.info(f"Test R2: {r2:.4f}")
    logger.info(f"Test MSE: {mse:.6f}")

    # Plot true vs predicted
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"True vs Predicted: {args.region_name} - {args.target}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
    # add R2 and MSE to the plot
    plt.text( 0.05, 0.95, f"R²: {r2:.4f}\nMSE: {mse:.6f}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Save the figure
    plot_path = os.path.join(results_dir, "true_vs_pred.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=1200)
    plt.close()

    logger.success(f"✅ True vs Predicted plot saved to {plot_path}")

    # Log plot to wandb
    wandb.log(
        {
            "test/test_r2": r2,
            "test/test_mse": mse,
            "test/true_vs_pred_plot": wandb.Image(plot_path),
        }
    )
    logger.success("✅ Test R2 and MSE logged to WandB")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrain GANDALF model from best sweep config"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--region_name", type=str, default="promoters_1024bp", help="Region name"
    )
    parser.add_argument(
        "--target", type=str, default="MLL-N", help="Target column name"
    )
    parser.add_argument(
        "--project", type=str, default="SEM_MLL-N_TF", help="WandB project name"
    )
    parser.add_argument(
        "--model_name", type=str, default="GANDALF_SEM", help="Model name"
    )
    parser.add_argument(
        "--best_config_path",
        type=str,
        required=True,
        help="Path to best_run_config.json from sweep",
    )

    args = parser.parse_args()

    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    results_dir = f"results/{args.project}/full_{args.model_name}_{args.region_name}_{args.target}_singlelabel_regression_{start_time}"

    main(args, results_dir)
