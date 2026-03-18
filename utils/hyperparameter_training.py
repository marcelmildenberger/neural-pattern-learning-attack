
import copy
import torch
from nepal.utils.pytorch_base_model import BaseModel
from nepal.utils.early_stopping import EarlyStopping
from nepal.utils.utils import calculate_performance_metrics, decode_labels_to_bi_grams, filter_high_scoring_bi_grams, load_experiment_datasets, map_probabilities_to_bi_grams, run_epoch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from ray import tune

# Define a function to train a model with a given configuration.
# This function is used by Ray Tune to train models with different hyperparameters.
def hyperparameter_training(config, data_dir, output_dim, alice_enc_hash, identifier, patience, min_delta, workers, ENC_CONFIG, NEPAL_CONFIG, GLOBAL_CONFIG, bi_gram_dict, all_bi_grams):
    # Sample all hyperparameters up front
    batch_size = int(config["batch_size"])
    num_layers = config["num_layers"]
    hidden_layer = config["hidden_layer"]
    dropout_rate = config["dropout_rate"]
    activation_fn = config["activation_fn"]
    loss_fn_name = config["loss_fn"]
    threshold = config["threshold"]
    optimizer_cfg = config["optimizer"]
    lr_scheduler_cfg = config["lr_scheduler"]

    # Load data
    datasets = load_experiment_datasets(data_dir, alice_enc_hash, identifier, ENC_CONFIG, NEPAL_CONFIG, GLOBAL_CONFIG, all_bi_grams, splits=("train", "val"))
    data_train, data_val = datasets["train"], datasets["val"]
    input_dim = data_train[0][0].shape[0]

    dataloader_train = DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=workers,
    )
    dataloader_val = DataLoader(
        data_val,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=workers,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BaseModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        hidden_layer=hidden_layer,
        dropout_rate=dropout_rate,
        activation_fn=activation_fn
    )
    model.to(device)

    # Loss function
    loss_functions = {
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
        "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss(),
        "SoftMarginLoss": nn.SoftMarginLoss(),
    }
    criterion = loss_functions[loss_fn_name]

    # Optimizer
    lr = optimizer_cfg["lr"].sample() if hasattr(optimizer_cfg["lr"], "sample") else optimizer_cfg["lr"]
    optimizer_name = optimizer_cfg["name"]
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        momentum = optimizer_cfg.get("momentum", 0.0)
        if hasattr(momentum, "sample"):
            momentum = momentum.sample()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Scheduler
    scheduler = None
    scheduler_name = lr_scheduler_cfg["name"]
    if scheduler_name == "StepLR":
        step_size = lr_scheduler_cfg["step_size"].sample() if hasattr(lr_scheduler_cfg["step_size"], "sample") else lr_scheduler_cfg["step_size"]
        gamma = lr_scheduler_cfg["gamma"].sample() if hasattr(lr_scheduler_cfg["gamma"], "sample") else lr_scheduler_cfg["gamma"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "ExponentialLR":
        gamma = lr_scheduler_cfg["gamma"].sample() if hasattr(lr_scheduler_cfg["gamma"], "sample") else lr_scheduler_cfg["gamma"]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_name == "ReduceLROnPlateau":
        factor = lr_scheduler_cfg["factor"].sample() if hasattr(lr_scheduler_cfg["factor"], "sample") else lr_scheduler_cfg["factor"]
        patience_sched = lr_scheduler_cfg["patience"].sample() if hasattr(lr_scheduler_cfg["patience"], "sample") else lr_scheduler_cfg["patience"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=lr_scheduler_cfg["mode"],
            factor=factor,
            patience=patience_sched
        )
    elif scheduler_name == "CosineAnnealingLR":
        T_max = lr_scheduler_cfg["T_max"].sample() if hasattr(lr_scheduler_cfg["T_max"], "sample") else lr_scheduler_cfg["T_max"]
        eta_min = lr_scheduler_cfg["eta_min"].sample() if hasattr(lr_scheduler_cfg["eta_min"], "sample") else lr_scheduler_cfg["eta_min"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name == "CyclicLR":
        base_lr = lr_scheduler_cfg["base_lr"].sample() if hasattr(lr_scheduler_cfg["base_lr"], "sample") else lr_scheduler_cfg["base_lr"]
        max_lr = lr_scheduler_cfg["max_lr"].sample() if hasattr(lr_scheduler_cfg["max_lr"], "sample") else lr_scheduler_cfg["max_lr"]
        step_size_up = lr_scheduler_cfg["step_size_up"].sample() if hasattr(lr_scheduler_cfg["step_size_up"], "sample") else lr_scheduler_cfg["step_size_up"]
        mode_cyclic = lr_scheduler_cfg["mode_cyclic"].sample() if hasattr(lr_scheduler_cfg["mode_cyclic"], "sample") else lr_scheduler_cfg["mode_cyclic"]
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode=mode_cyclic,
            cycle_momentum=False
        )
    elif scheduler_name == "None":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    # Initialize variables for tracking best validation loss, model state, performance metrics, and early stopping
    best_val_loss = float('inf')
    best_model_state = None
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_dice = 0.0
    total_val_loss = 0.0
    num_samples = 0
    epochs = 0
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    # Train the model for a specified number of epochs.
    for _ in range(NEPAL_CONFIG["Epochs"]):
        epochs += 1
        train_loss = run_epoch(
            model, dataloader_train, criterion, optimizer, device,
            is_training=True, verbose=GLOBAL_CONFIG["Verbose"], scheduler=scheduler
        )
        val_loss = run_epoch(
            model, dataloader_val, criterion, optimizer, device,
            is_training=False, verbose=GLOBAL_CONFIG["Verbose"], scheduler=scheduler
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        total_val_loss += val_loss
        if early_stopper(val_loss):
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    for data, labels, _ in dataloader_val:
        actual_bi_grams = decode_labels_to_bi_grams(bi_gram_dict, labels)
        data = data.to(device)
        logits = model(data)
        probabilities = torch.sigmoid(logits)
        batch_bi_gram_scores = map_probabilities_to_bi_grams(bi_gram_dict, probabilities)
        batch_filtered_bi_gram_scores = filter_high_scoring_bi_grams(batch_bi_gram_scores, threshold)
        dice, precision, recall, f1 = calculate_performance_metrics(
            actual_bi_grams, batch_filtered_bi_gram_scores)
        total_dice += dice
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        num_samples += data.size(0)
    metrics = {
            "average_dice": total_dice / num_samples,
            "average_precision": total_precision / num_samples,
            "average_recall": total_recall / num_samples,
            "average_f1": total_f1 / num_samples,
            "total_val_loss": total_val_loss,
            "len_train": len(dataloader_train.dataset),
            "len_val": len(dataloader_val.dataset),
            "epochs": epochs
    }
    tune.report(metrics)
