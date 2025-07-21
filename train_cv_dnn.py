import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
import numpy as np
import matplotlib.pyplot as plt
import optuna
import json
import os
import random
from datetime import datetime
from optuna.samplers import TPESampler

from mymodelzoo.dnn import DynamicDNN # Import from your new library

# Import for Optuna visualizations
try:
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_pareto_front
    OPTUNA_VIZ_INSTALLED = True
except ImportError:
    OPTUNA_VIZ_INSTALLED = False

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Trainer Class ---
class PyTorchTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    def train_epoch(self, data_loader):
        self.model.train()
        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        for features, labels in data_loader:
            features, labels = features.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        return running_loss / total_samples, correct_predictions / total_samples

    def validate_epoch(self, data_loader):
        self.model.eval()
        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in data_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        return running_loss / total_samples, correct_predictions / total_samples

    def train(self, train_loader, val_loader, num_epochs, patience, scheduler=None, verbose=False):
        self.history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        best_val_loss = np.inf
        patience_counter = 0
        best_model_state = None
        best_val_acc_at_best_loss = 0.0
        train_acc_at_best_loss = 0.0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc_at_best_loss = val_acc
                train_acc_at_best_loss = train_acc
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                if verbose and (epoch + 1) % 5 == 0:
                     print(f"  Epoch {epoch+1:03d}/{num_epochs} | New best! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\n--- Early stopping triggered at epoch {epoch+1} ---")
                    break
            if scheduler:
                scheduler.step()

        if best_model_state:
            if verbose:
                print(f"\nLoading best model with Val Acc: {best_val_acc_at_best_loss:.4f}")
            self.model.load_state_dict(best_model_state)
        
        return best_val_acc_at_best_loss, train_acc_at_best_loss

    def evaluate(self, test_loader, class_names=None):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        print("\n--- Final Model Evaluation ---")
        print(classification_report(all_preds, all_labels, target_names=class_names, zero_division=0))

# --- Data Preparation Function ---
def prepare_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72125, stratify=y)
    return X_train, y_train, X_test, y_test, iris.target_names

# --- Optuna Objective Function ---
def objective(trial, X_train_full, y_train_full, input_size, num_classes, num_epochs, device, cv_method='kfold', n_splits = 5):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_size = trial.suggest_int('hidden_size', 8, 32, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    activation_name = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh', 'ELU'])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    if cv_method == 'kfold':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=72125)
    elif cv_method == 'loocv':
        kf = LeaveOneOut()
    else:
        raise ValueError("Invalid cv_method. Choose 'kfold' or 'loocv'.")

    fold_accuracies, fold_gaps = [], []

    for train_index, val_index in kf.split(X_train_full, y_train_full):
        X_train, X_val = X_train_full[train_index], X_train_full[val_index]
        y_train, y_val = y_train_full[train_index], y_train_full[val_index]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=batch_size)

        activation_fn = getattr(nn, activation_name)
        model = DynamicDNN(input_size, hidden_size, num_classes, n_layers, activation_fn, dropout_rate)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        trainer = PyTorchTrainer(model, nn.CrossEntropyLoss(), optimizer, device)
        
        val_acc, train_acc = trainer.train(train_loader, val_loader, num_epochs, patience=10, verbose=False)
        fold_accuracies.append(val_acc)
        fold_gaps.append(abs(train_acc - val_acc))

    mean_accuracy = np.mean(fold_accuracies)
    std_dev = np.std(fold_accuracies)
    mean_gap = np.mean(fold_gaps)
    
    return mean_accuracy, std_dev, mean_gap

# --- Plotting Function ---
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['train_loss'], label='Train Loss'); ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_accuracy'], label='Train Accuracy'); ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(True)
    plt.show()

# --- Callback for printing progress ---
def print_progress(study, trial):
    print(f"Trial {trial.number} finished.")
    if trial.state == optuna.trial.TrialState.COMPLETE:
        acc, std, gap = trial.values
        print(f"  Values: (Acc: {acc:.4f}, Std Dev: {std:.4f}, Gap: {gap:.4f})")
    if study.best_trials:
        print(f"  Current Pareto front size: {len(study.best_trials)}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    LOAD_PARAMS_FROM_FILE = True
    CV_METHOD = 'kfold' # Options: 'kfold' or 'loocv'
    N_SPLITS = 5
    PARAMS_PREFIX = "outputs/best_hyperparameters_cv"
    MODEL_PREFIX = "models/best_model_cv"
    LATEST_SUFFIX = "_latest"
    
    N_TRIALS = 50
    NUM_EPOCHS_OPTUNA = 50
    NUM_EPOCHS_FINAL = 200
    FINAL_MODEL_PATIENCE = 15
    STABILITY_THRESHOLD = 0.05
    GENERALIZATION_GAP_THRESHOLD = 0.10

    set_seed(42)

    os.makedirs(os.path.dirname(PARAMS_PREFIX), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PREFIX), exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    X_train_full, y_train_full, X_test, y_test, class_names = prepare_data()
    input_size, num_classes = X_train_full.shape[1], len(set(y_train_full))
    
    best_params = {}
    params_file_to_load = f"{PARAMS_PREFIX}{LATEST_SUFFIX}.json"
    # In your main block, before creating the study
    sampler = TPESampler(n_startup_trials = N_TRIALS // 2)


    if LOAD_PARAMS_FROM_FILE and os.path.exists(params_file_to_load):
        print(f"--- Loading hyperparameters from {params_file_to_load} ---")
        with open(params_file_to_load, 'r') as f:
            best_params = json.load(f)
        print("Parameters loaded successfully.")
    else:
        print(f"\n--- Starting Optuna search with {N_TRIALS} trials (CV Method: {CV_METHOD}) ---")
        study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'], sampler=sampler)
        study.optimize(
            lambda trial: objective(trial, X_train_full, y_train_full, input_size, num_classes, NUM_EPOCHS_OPTUNA, DEVICE, cv_method=CV_METHOD, n_splits=N_SPLITS),
            n_trials=N_TRIALS,
            callbacks=[print_progress]
        )

        print("\n--- Optuna search complete ---")

        if OPTUNA_VIZ_INSTALLED:
            print("\n--- Generating Optuna Visualizations ---")
            plot_pareto_front(study, target_names=["Mean CV Accuracy", "Std Dev", "Mean Gap"]).show()
            plot_param_importances(study, target=lambda t: t.values[0], target_name="Mean CV Accuracy").show()
        else:
            print("\nSkipping visualizations: `plotly` is not installed.")
        
        print(f"Found {len(study.best_trials)} optimal trials on the Pareto front.")
        
        stable_trials = [t for t in study.best_trials if t.values[1] < STABILITY_THRESHOLD]
        if not stable_trials:
            stable_trials = study.best_trials

        generalizing_trials = [t for t in stable_trials if t.values[2] < GENERALIZATION_GAP_THRESHOLD]
        if not generalizing_trials:
            generalizing_trials = stable_trials

        final_candidates = sorted(generalizing_trials, key=lambda t: t.values[0], reverse=True)

        if not final_candidates:
            print("No suitable trials were found.")
            exit()
        
        print("\n--- Top 5 Candidate Trials ---")
        for i, trial in enumerate(final_candidates[:5]):
            acc, std, gap = trial.values
            print(f"\n{i+1}. Trial #{trial.number}")
            print(f"   - Mean CV Accuracy:      {acc:.4f}")
            print(f"   - Std Dev:               {std:.4f}")
            print(f"   - Mean Generalization Gap: {gap:.4f}")
            print(f"   - Params: {trial.params}")

        best_trial = final_candidates[0]
        best_params = best_trial.params
        print("\n--- Automatically proceeding with the #1 best trial ---")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timestamped_params_file = f"{PARAMS_PREFIX}_{timestamp}.json"
        latest_params_file = f"{PARAMS_PREFIX}{LATEST_SUFFIX}.json"

        run_data_to_save = {
            "best_trial_number": best_trial.number,
            "objectives": {
                "mean_cv_accuracy": best_trial.values[0],
                "std_dev": best_trial.values[1],
                "mean_gap": best_trial.values[2]
            },
            "hyperparameters": best_params
        }

        print(f"\n--- Saving detailed results to {timestamped_params_file} ---")
        with open(timestamped_params_file, 'w') as f:
            json.dump(run_data_to_save, f, indent=4)
        
        with open(latest_params_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"Parameters for latest run saved to {latest_params_file}")
    
    # --- Train Final Model using Best Parameters ---
    print("\nBest hyperparameters for this run:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
    
    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_full)
    X_test_final = scaler.transform(X_test)
    
    best_batch_size = best_params['batch_size']
    
    X_train_final_split, X_val_final_split, y_train_final_split, y_val_final_split = train_test_split(
        X_train_final, y_train_full, test_size=0.2, random_state=72125, stratify=y_train_full
    )
    
    final_train_loader = DataLoader(TensorDataset(torch.tensor(X_train_final_split, dtype=torch.float32), torch.tensor(y_train_final_split, dtype=torch.long)), batch_size=best_batch_size, shuffle=True)
    final_val_loader = DataLoader(TensorDataset(torch.tensor(X_val_final_split, dtype=torch.float32), torch.tensor(y_val_final_split, dtype=torch.long)), batch_size=best_batch_size)
    final_test_loader = DataLoader(TensorDataset(torch.tensor(X_test_final, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=best_batch_size)

    final_model = DynamicDNN(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['n_layers'],
        activation_fn=getattr(nn, best_params['activation']),
        dropout_rate=best_params['dropout_rate']
    )
    final_optimizer = getattr(optim, best_params['optimizer'])(final_model.parameters(), lr=best_params['lr'])
    
    final_trainer = PyTorchTrainer(final_model, nn.CrossEntropyLoss(), final_optimizer, DEVICE)

    print(f"\n--- Training final model on full training data ---")
    final_trainer.train(final_train_loader, final_val_loader, NUM_EPOCHS_FINAL, patience=FINAL_MODEL_PATIENCE, verbose=True)
    
    final_trainer.evaluate(final_test_loader, class_names=class_names)
    
    plot_history(final_trainer.history)

    # --- Save the Final Trained Model ---
    if 'timestamp' not in locals():
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_filename_ts = f"{MODEL_PREFIX}_{timestamp}.pth"
    model_filename_latest = f"{MODEL_PREFIX}{LATEST_SUFFIX}.pth"
    
    print(f"\n--- Saving final model state to {model_filename_ts} ---")
    torch.save(final_trainer.model.state_dict(), model_filename_ts)
    torch.save(final_trainer.model.state_dict(), model_filename_latest)
    print(f"Model for latest run also saved to {model_filename_latest}")