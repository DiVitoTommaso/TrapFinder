from models import *
from losses import *
from dataset import *
from environment import *

# ----------------------------
# AMP scaler for mixed precision
# ----------------------------
scaler = torch.cuda.amp.GradScaler()


# ----------------------------
# Training
# ----------------------------
def train_epoch(model, dataloader, optimizer):
    # Set model to training mode (enables dropout, batchnorm, etc.)
    model.train()

    # Use Hungarian set loss to handle unordered point predictions
    criterion = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, potentials=False)
    running_loss = 0.0

    # Iterate over batches from the dataloader
    for images, centers in dataloader:
        # Move images and ground-truth centers to the correct device (GPU/CPU)
        images, centers = images.to(DEVICE), centers.to(DEVICE)

        # Reset gradients from previous step
        optimizer.zero_grad()

        # Mixed-precision context for faster computation
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            # Forward pass: predict points
            outputs = model(images)

            # Convert predictions and targets to float32 for Hungarian loss / distance computation
            outputs = outputs.float()
            centers = centers.float()

            # Compute the loss between predicted points and ground truth
            loss = criterion(outputs, centers).mean()

        # Backward pass with gradient scaling (for mixed precision)
        scaler.scale(loss).backward()

        # Gradient clipping to prevent exploding gradients
        scaler.unscale_(optimizer)  # unscale gradients before clipping
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        # Optimizer step and update scaler for mixed precision
        scaler.step(optimizer)
        scaler.update()

        # Accumulate total loss (weighted by batch size)
        running_loss += loss.item() * images.size(0)

    # Return average loss over the entire dataset
    return running_loss / len(dataloader.dataset)


# ----------------------------
# Evaluation
# ----------------------------
def eval_epoch(model, dataloader):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, centers in dataloader:
            images, centers = images.to(DEVICE), centers.to(DEVICE)
            outputs = model(images)  # (B, N, 2)

            B, N, _ = outputs.shape
            batch_loss = 0.0

            for b in range(B):
                # Calcola la matrice dei costi tra punti predetti e ground truth
                cost_matrix = torch.cdist(outputs[b], centers[b], p=2).cpu().numpy()

                # Risolvi matching ottimale (Hungarian)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                # Punti abbinati
                matched_preds = outputs[b, row_ind]
                matched_targets = centers[b, col_ind]

                # Loss = media distanza euclidea per campione
                sample_loss = torch.norm(matched_preds - matched_targets, dim=-1).mean()
                batch_loss += sample_loss.item()

            running_loss += batch_loss

    # Media su tutto il dataset
    return running_loss / len(dataloader.dataset)


# ----------------------------
# Train model
# ----------------------------
def train_model(train_loader, val_loader, num_restarts, epochs, lr, patience,
                weight_decay, hidden_dim, num_layers, drop_prob, drop_path_prob=0):
    """
    Train a ResNet model with multiple restarts, early stopping, and LR scheduling.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_restarts: Number of times to restart training from scratch
        epochs: Maximum epochs per restart
        lr: Initial learning rate
        patience: Early stopping patience (epochs without improvement)
        weight_decay: L2 regularization factor
        hidden_dim: Hidden dimension of ResNet
        num_layers: Number of ResNet layers
        drop_prob: Dropout probability
        drop_path_prob: Drop path probability (optional)
    Returns:
        best_model: ResNet model with lowest validation loss across restarts
        best_val_loss: Corresponding validation loss
    """

    # Initialize variables to track the best model across all restarts
    best_model_state = None
    best_val_loss = float("inf")

    # Repeat training with different random initializations (restarts)
    for restart in range(num_restarts):
        # Initialize a new model
        model = ResNet(num_points=MAX_POINTS, hidden_dim=hidden_dim,
                       num_layers=num_layers, drop_prob=drop_prob, img_size=INPUT_SIZE).to(DEVICE)

        # Optimizer with AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Reduce LR on plateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # minimize validation loss
            factor=0.8,  # reduce LR by 20% when plateau
            patience=10,  # wait 10 epochs before reducing
            min_lr=1e-6  # minimum learning rate
        )

        # Track early stopping within this restart
        epochs_no_improve = 0
        restart_best_val = float("inf")
        restart_best_state = None

        # Progress bar for the current restart
        with tqdm(total=epochs, desc=f"Try {restart + 1}") as pbar:
            for epoch in range(epochs):
                # --- Training ---
                train_loss = train_epoch(model, train_loader, optimizer)

                # --- Validation ---
                val_loss = eval_epoch(model, val_loader)

                # Update progress bar with metrics
                pbar.set_postfix({
                    "Train Loss": f"{train_loss:.4f}",
                    "Val Loss": f"{val_loss:.4f}",
                    "LR": optimizer.param_groups[0]['lr']
                })

                # Update learning rate scheduler based on validation loss
                scheduler.step(val_loss)

                # --- Early stopping logic ---
                if val_loss < restart_best_val:
                    # Validation improved → save model state
                    restart_best_val = val_loss
                    restart_best_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    # No improvement → increment counter
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        # Stop early if patience exceeded
                        break

                # Advance progress bar
                pbar.update(1)

        # After this restart, update best model across all restarts
        if restart_best_val < best_val_loss:
            best_val_loss = restart_best_val
            best_model_state = restart_best_state

    # Load the best model weights into a fresh model instance
    best_model = ResNet(num_points=MAX_POINTS, hidden_dim=hidden_dim,
                        num_layers=num_layers, drop_prob=drop_prob, img_size=INPUT_SIZE).to(DEVICE)
    best_model.load_state_dict(best_model_state)

    # Return best model and its validation loss
    return best_model, best_val_loss


# ----------------------------
# Main random search + KFold
# ----------------------------
if __name__ == "__main__":
    # -------------------------------
    # Hyperparameters and settings
    # -------------------------------
    k_folds = 4  # Number of folds for cross-validation
    n_trials = 1  # Number of random hyperparameter trials
    batch_size = 8  # Batch size for training

    save_path = "Top10_hparams.json"  # JSON file to store top hyperparameter results
    models_dir = f"{LABEL.lower().replace('-', '')}_models"  # Directory to save models

    # Initialize results list, optionally load previous top results
    results = []
    os.makedirs(models_dir, exist_ok=True)
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} previous top results from {save_path}")

    # -------------------------------
    # Dataset initialization
    # -------------------------------
    dataset_pure = CenterRegressionDataset(ROOT_DIR, LABEL, MAX_POINTS, INPUT_SIZE)  # clean dataset
    dataset_aug = CenterRegressionDataset(ROOT_DIR, LABEL, MAX_POINTS, INPUT_SIZE)  # augmented dataset

    # -------------------------------
    # Random hyperparameter search
    # -------------------------------
    for trial in range(n_trials):
        # Randomly sample hyperparameters
        lr = random.uniform(1e-4, 1e-4)  # learning rate (here fixed)
        wd = random.uniform(1e-12, 1e-12)  # weight decay (here fixed)
        drop_prob = random.uniform(0.05, 0.05)  # dropout probability (here fixed)
        hidden_dim = random.randint(128, 128)  # hidden dimension (here fixed)
        num_layers = random.randint(6, 6)  # number of layers (here fixed)

        fold_losses = []  # to store validation losses per fold
        fold_model_paths = []  # to store paths of trained models

        # K-Fold cross-validation
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=1804)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset_pure)):
            # --- Train/val split ---
            dataset_aug.disable_augmentation()
            train_subset_base = Subset(dataset_aug, train_idx)  # training subset with augmentation
            val_subset = Subset(dataset_pure, val_idx)  # validation subset (clean)

            # --- Oversample the training subset ---
            train_subset = smoter_oversample(
                train_subset_base,
                num_samples=len(train_subset_base),  # number of synthetic samples
                sigma=0.01  # small Gaussian noise
            )

            # --- Sampler and DataLoader ---
            sampler = make_weighted_sampler(train_subset, 12, 15)  # optional weighted sampling
            dataset_aug.enable_augmentation()  # enable online augmentation

            # DataLoaders
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                sampler=sampler
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=True
            )

            # -------------------------------
            # Train the model for this fold
            # -------------------------------
            model, best_val_loss = train_model(
                train_loader, val_loader,
                num_restarts=2,  # multiple restarts for better minima
                patience=50,  # early stopping patience
                epochs=300,  # max epochs
                weight_decay=wd,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                drop_prob=drop_prob,
                lr=lr
            )
            fold_losses.append(best_val_loss)

            # Save trained model for this fold
            model_path = os.path.join(models_dir, f"Model{trial + 1}-Fold{fold + 1}.pt")
            torch.save(model.state_dict(), model_path)
            fold_model_paths.append(model_path)

        # -------------------------------
        # Evaluate this hyperparameter trial
        # -------------------------------
        mean_loss = float(np.mean(fold_losses))
        std_loss = float(np.std(fold_losses))
        score = mean_loss + std_loss  # simple metric combining mean + variability

        # Store trial results
        results.append({
            "score": score,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "lr": lr,
            "weight_decay": wd,
            "hidden_dim": hidden_dim,
            "drop_prob": drop_prob,
            "num_layers": num_layers,
            "fold_models": fold_model_paths
        })

        # Keep only top-10 trials by score
        results = sorted(results, key=lambda x: x["score"])[:5]

        # Save JSON summary (without large model files)
        json_safe = [{k: v for k, v in r.items() if k != "fold_models"} for r in results]
        with open(save_path, "w") as f:
            json.dump(json_safe, f, indent=2)

        # Print summary of current top-10
        print("\nCurrent Top-5:")
        for i, r in enumerate(results, 1):
            print(f"Top {i}: Score={r['score']:.4f}, Mean={r['mean_loss']:.4f}, Std={r['std_loss']:.4f}, "
                  f"LR={r['lr']:.6f}, WD={r['weight_decay']:.6f}, "
                  f"hidden_dim={r['hidden_dim']}, num_layers={r['num_layers']}, drop_prob={r['drop_prob']}")
