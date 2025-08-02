import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import copy
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import json
import os
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from mymodelzoo.cnn1D import DynamicCNN

try:
    from optuna.visualization import plot_param_importances, plot_pareto_front
    OPTUNA_VIZ_INSTALLED = True
except ImportError:
    OPTUNA_VIZ_INSTALLED = False

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def define_hyperparameters(trial: optuna.trial.Trial):
    params = {}
    params['activation_name'] = trial.suggest_categorical('activation_name', ['ReLU', 'GELU', 'SiLU'])
    params['n_conv_blocks'] = trial.suggest_int('n_conv_blocks', 3, 6)
    params['base_channels'] = trial.suggest_int('base_channels', 24, 64, log=True)
    params['block_conv_layers'] = trial.suggest_int('block_conv_layers', 1, 2)
    params['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5])
    params['n_fc_layers'] = trial.suggest_int('n_fc_layers', 1, 2)
    params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5)
    params['fc_size'] = trial.suggest_int('fc_size', 64, 256, log=True)
    params['optimizer_name'] = trial.suggest_categorical('optimizer_name', ['AdamW', 'Adam'])
    params['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    params['scheduler_name'] = trial.suggest_categorical('scheduler_name', ['StepLR', 'CosineAnnealingLR'])
    if params['scheduler_name'] == 'StepLR':
        params['scheduler_step_size'] = trial.suggest_int('scheduler_step_size', 5, 15)
        params['scheduler_gamma'] = trial.suggest_float('scheduler_gamma', 0.1, 0.9)
    else:
        params['scheduler_t_max'] = trial.suggest_int('scheduler_t_max', 10, 50)
    params['batch_size'] = trial.suggest_categorical('batch_size', [64, 128])
    return params


# --- TRAINER CLASS ---
# --- TRAINER CLASS ---
class PyTorchTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [], 'lr': []}

    def train_epoch(self, data_loader):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        for features, labels in data_loader:
            features = features.to(self.device, dtype=torch.float)
            labels = labels.to(self.device, dtype=torch.long)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return running_loss / total, correct / total

    def validate_epoch(self, data_loader):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device, dtype=torch.float)
                labels = labels.to(self.device, dtype=torch.long)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss / total, correct / total

    def train(self, train_loader, val_loader, num_epochs, patience, warmup_epochs=5, verbose=False):
        best_val_loss, patience_counter, best_model_state = float('inf'), 0, None
        best_val_acc, train_acc_at_best = 0.0, 0.0
        
        # --- Learning Rate Warm-up Logic ---
        base_lr = self.optimizer.param_groups[0]['lr']
        warmup_start_lr = 1e-6

        for epoch in range(num_epochs):
            # --- Warm-up phase ---
            if epoch < warmup_epochs:
                # Linearly increase the learning rate
                lr = warmup_start_lr + (base_lr - warmup_start_lr) * (epoch / warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            # --- Post-warm-up phase ---
            elif epoch == warmup_epochs:
                # Restore the base learning rate so the scheduler can take over
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = base_lr
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Only step the main scheduler after the warm-up is complete
            if self.scheduler and epoch >= warmup_epochs:
                self.scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                train_acc_at_best = train_acc 
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose: print(f"\n--- Early stopping at epoch {epoch+1} ---")
                    break
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        return best_val_acc, train_acc_at_best

    def evaluate(self, test_loader, class_names=None):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device, dtype=torch.float)
                labels = labels.to(self.device, dtype=torch.long)
                outputs = self.model(features)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        print("\n--- Final Model Evaluation on Test Set ---")
        print(classification_report(all_preds, all_preds, target_names=class_names, zero_division=0))

def prepare_data(x_path="../Data-Generation/outputs/cnn_ready_data_X.npy", y_path="../Data-Generation/outputs/cnn_ready_data_y.npy"):
    """Loads the pre-processed EEG data, splits it, and creates PyTorch Datasets."""
    print("--- Loading and preparing custom EEG data ---")
    try:
        X = np.load(x_path)
        y = np.load(y_path)
    except FileNotFoundError:
        print(f"Error: Make sure '{x_path}' and '{y_path}' are in the correct directory.")
        exit()
        
    print(f"Loaded data shapes: X={X.shape}, y={y.shape}")

    # Split data into training (80%) and testing (20%) sets
    # Stratify ensures the class distribution is the same in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=73125, stratify=y
    )
    print("Saving test split to files for XAI analysis...")
    np.save("test data/X_test.npy", X_test)
    np.save("test data/y_test.npy", y_test)
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create TensorDataset objects
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Define the class names based on the converter script's output
    # IMPORTANT: This order must match the label encoding from the converter
    class_names = ['Brain Rot', 'Brain Tumor', 'Metabolic Encephalopathy', 'Normal', 'Sleep Disorder']
    
    print("Data preparation complete.")
    return train_dataset, test_dataset, class_names

def plot_history(history):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 5))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_accuracy'], label='Train Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(history['lr'], label='Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.legend()
    ax3.grid(True)

    plt.show()

# --- OBJECTIVE FUNCTION ---
def objective(trial, full_train_dataset, input_shape, num_classes, num_epochs, device, n_splits):
    params = define_hyperparameters(trial)
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=73125)
    targets = full_train_dataset.tensors[1]
    fold_val_accuracies, fold_train_accuracies = [], []

    for train_index, val_index in kf.split(np.zeros(len(targets)), targets):
        train_loader = DataLoader(Subset(full_train_dataset, train_index), batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(Subset(full_train_dataset, val_index), batch_size=params['batch_size'])
        
        fold_model = DynamicCNN(params, input_shape, num_classes)
        fold_optimizer = getattr(optim, params['optimizer_name'])(fold_model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        fold_scheduler = optim.lr_scheduler.CosineAnnealingLR(fold_optimizer, T_max=num_epochs)

        trainer = PyTorchTrainer(fold_model, nn.CrossEntropyLoss(), fold_optimizer, fold_scheduler, device)
        val_acc, train_acc = trainer.train(train_loader, val_loader, num_epochs, patience=10, warmup_epochs= 5)
        fold_val_accuracies.append(val_acc)
        fold_train_accuracies.append(train_acc)
    
    # Calculate all three objectives
    mean_val_acc = np.mean(fold_val_accuracies)
    std_val_acc = np.std(fold_val_accuracies)
    mean_gap = np.mean([abs(t - v) for t, v in zip(fold_train_accuracies, fold_val_accuracies)])
    
    return mean_val_acc, std_val_acc, mean_gap

def print_progress(study, trial):
    print(f"Trial {trial.number} finished.")
    if trial.state == optuna.trial.TrialState.COMPLETE:
        acc, std, gap = trial.values
        print(f"  Values: (Acc: {acc:.4f}, Std: {std:.4f}, Gap: {gap:.4f})")
    if study.best_trials:
        print(f"  Current Pareto front size: {len(study.best_trials)}")
        
        # Apply the same selection logic to find the current best candidate
        stable_trials = [t for t in study.best_trials if t.values[1] < 0.05]
        if not stable_trials: stable_trials = study.best_trials
        
        generalizing_trials = [t for t in stable_trials if t.values[2] < 0.10]
        if not generalizing_trials: generalizing_trials = stable_trials

        best_so_far = max(generalizing_trials, key=lambda t: t.values[0])
        
        best_acc, best_std, best_gap = best_so_far.values
        print(f"  Best trial so far (#{best_so_far.number}): Acc: {best_acc:.4f}, Std: {best_std:.4f}, Gap: {best_gap:.4f}")



if __name__ == "__main__":
    LOAD_PARAMS_FROM_FILE = True
    PARAMS_PREFIX = "outputs/nas_best_hyperparameters_3obj_1D_2Feature_Signal_Second_Pass_100_Trials"
    MODEL_PREFIX = "models/nas_best_model_3obj_1D_2Feature_Signal_Second_Pass_100_Trials"
    
    N_TRIALS = 100
    NUM_EPOCHS_OPTUNA = 50
    N_SPLITS_CV = 5
    NUM_EPOCHS_FINAL = 50
    FINAL_MODEL_PATIENCE = 10

    set_seed(73125)
    os.makedirs(os.path.dirname(PARAMS_PREFIX), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PREFIX), exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE} 🚀")

    # --- UPDATED DATA LOADING ---
    train_dataset, test_dataset, class_names = prepare_data()
    
    # Dynamically get input shape and number of classes from the loaded data
    # Shape from loader: (N, Timesteps, Features)
    # Shape for Conv1d: (Features, Timesteps)
    sample_x, _ = train_dataset[0]
    num_timesteps, num_features = sample_x.shape
    input_shape = (num_features, num_timesteps) 
    num_classes = len(class_names)
    print(f"Input shape for CNN: {input_shape} | Number of classes: {num_classes}")
    
    params_file_to_load = f"{PARAMS_PREFIX}_latest.json"

    if LOAD_PARAMS_FROM_FILE and os.path.exists(params_file_to_load):
        print(f"--- Loading results from {params_file_to_load} ---")
        with open(params_file_to_load, 'r') as f:
            saved_results = json.load(f)
        best_params = saved_results['hyperparameters']
        print("Objectives from loaded file:")
        for key, value in saved_results['objectives'].items():
            print(f"  - {key}: {value:.4f}")
    else:
        print(f"\n--- Starting Optuna NAS with {N_TRIALS} trials ---")
        sampler = TPESampler(n_startup_trials=max(1, N_TRIALS // 2), seed=73125)
        
        study = optuna.create_study(
            directions=['maximize', 'minimize', 'minimize'],
            sampler=sampler
        )
        print(f"Optimizing for: [Maximize Acc, Minimize Std, Minimize Gap]")
        
        study.optimize(
            lambda trial: objective(trial, train_dataset, input_shape, num_classes, NUM_EPOCHS_OPTUNA, DEVICE, n_splits=N_SPLITS_CV),
            n_trials=N_TRIALS,
            callbacks=[print_progress]
        )

        print("\n--- Optuna search complete ---")

        if OPTUNA_VIZ_INSTALLED:
            plot_pareto_front(study, target_names=["Mean Acc", "Std Dev", "Train-Val Gap"]).show()
            plot_param_importances(study, target=lambda t: t.values[0], target_name="Mean Acc").show()

        print("\n--- Selecting Best Trial from Pareto Front ---")
        stable_trials = [t for t in study.best_trials if t.values[1] < 0.05]
        if not stable_trials: stable_trials = study.best_trials
        generalizing_trials = [t for t in stable_trials if t.values[2] < 0.10]
        if not generalizing_trials: generalizing_trials = stable_trials
        best_trial = max(generalizing_trials, key=lambda t: t.values[0])
        best_params = best_trial.params
        
        print("\n--- Best Trial Selected ---")
        acc, std, gap = best_trial.values
        print(f"  - Trial Number: {best_trial.number}")
        print(f"  - Mean CV Accuracy: {acc:.4f}")
        print(f"  - Std Dev:          {std:.4f}")
        print(f"  - Train-Val Gap:    {gap:.4f}")
        results_to_save = {
            "objectives": {"mean_cv_accuracy": acc, "std_dev": std, "mean_gap": gap},
            "hyperparameters": best_params
        }
        
        with open(f"{params_file_to_load}", 'w') as f: json.dump(results_to_save, f, indent=4)
        print(f"\nParameters for best run saved to {params_file_to_load}")
    
    print("\n--- Training final model with best discovered architecture ---")
    final_model = DynamicCNN(best_params, input_shape, num_classes)
    
    # For the final run, split the full training data into a new train and validation set
    train_indices, val_indices = train_test_split(
        list(range(len(train_dataset))), 
        test_size=0.15, 
        stratify=train_dataset.tensors[1].numpy(), 
        random_state=73125
    )
    final_train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=best_params['batch_size'], shuffle=True)
    final_val_loader = DataLoader(Subset(train_dataset, val_indices), batch_size=best_params['batch_size'])
    final_test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

    final_optimizer = getattr(optim, best_params['optimizer_name'])(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    if best_params['scheduler_name'] == 'StepLR':
        final_scheduler = optim.lr_scheduler.StepLR(final_optimizer, step_size=best_params['scheduler_step_size'], gamma=best_params['scheduler_gamma'])
    else:
        final_scheduler = optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=NUM_EPOCHS_FINAL)

    final_trainer = PyTorchTrainer(final_model, nn.CrossEntropyLoss(), final_optimizer, final_scheduler, DEVICE)
    final_trainer.train(final_train_loader, final_val_loader, NUM_EPOCHS_FINAL, patience=FINAL_MODEL_PATIENCE,warmup_epochs=5, verbose=True)
    
    plot_history(final_trainer.history)
    final_trainer.evaluate(final_test_loader, class_names=class_names)
    
    torch.save(final_trainer.model.state_dict(), f"{MODEL_PREFIX}_latest.pth")
    print(f"\n--- Final model saved to {MODEL_PREFIX}_latest.pth ---")