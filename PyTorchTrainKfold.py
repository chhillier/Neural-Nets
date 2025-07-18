import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# --- Model Definition ---
class DynamicDNN(nn.Module):
    """
    A dynamically built deep neural network using nn.Sequential.
    """
    def __init__(self, input_size, hidden_size, num_classes, num_layers, activation_fn, dropout_rate):
        """
        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in each hidden layer.
            num_classes (int): The number of output classes.
            num_layers (int): The total number of hidden layers.
            activation_fn (torch.nn.Module): The activation function to use.
            dropout_rate (float): The dropout rate to apply after each hidden layer.
        """
        super(DynamicDNN, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation_fn())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
            
        # Output layer
        layers.append(nn.Linear(hidden_size, num_classes))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Defines the forward pass of the model."""
        return self.model(x)

# --- Trainer Class ---
class PyTorchTrainer:
    """
    A generic trainer class for PyTorch models, handling training, validation,
    evaluation, and early stopping.
    """
    def __init__(self, model, criterion, optimizer, device):
        """
        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            criterion: The loss function.
            optimizer: The optimization algorithm.
            device (torch.device): The device to train on (e.g., 'cpu', 'cuda').
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    def train_epoch(self, data_loader):
        """Performs a single training epoch."""
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
        """Performs a single validation epoch."""
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

    def train(self, train_loader, val_loader, num_epochs, patience, scheduler=None, verbose=True):
        """
        Runs the full training loop with early stopping and checkpointing.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            num_epochs (int): The maximum number of epochs to train.
            patience (int): How many epochs to wait for improvement before stopping.
            scheduler: An optional learning rate scheduler.
            verbose (bool): If True, prints training progress.

        Returns:
            float: The validation accuracy of the best model found.
        """
        self.history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        best_val_loss = np.inf
        patience_counter = 0
        best_model_state = None
        best_accuracy_at_best_loss = 0.0

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            # Early stopping logic (monitors validation loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_accuracy_at_best_loss = val_acc
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                if verbose:
                    print(f"  Epoch {epoch+1:03d}/{num_epochs} | New best model! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\n--- Early stopping triggered at epoch {epoch+1} ---")
                    break
            
            if scheduler:
                scheduler.step()

        # Load the best model state found during training
        if best_model_state:
            if verbose:
                print(f"\nLoading best model (Epoch {epoch+1-patience_counter}) with Val Loss: {best_val_loss:.4f} and Val Acc: {best_accuracy_at_best_loss:.4f}")
            self.model.load_state_dict(best_model_state)
        
        return best_accuracy_at_best_loss

    def evaluate(self, test_loader, class_names=None):
        """Evaluates the final model on the test set."""
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
    """Loads the Iris dataset and splits it into training and test sets."""
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, y_train, X_test, y_test, iris.target_names

# --- Optuna Objective Function ---
def objective(trial, X_train_full, y_train_full, input_size, num_classes, num_epochs, device, cv_method='kfold'):
    """
    Optuna objective function to find the best hyperparameters for the DNN.
    This function defines the search space and performs cross-validation.
    """
    # Define the hyperparameter search space
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_size = trial.suggest_int('hidden_size', 8, 16, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7, log=True)
    activation_name = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh', 'ELU', 'PReLU'])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Select the cross-validation iterator based on the chosen method
    if cv_method == 'kfold':
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    elif cv_method == 'loocv':
        kf = LeaveOneOut()
    else:
        raise ValueError("Invalid cv_method. Choose 'kfold' or 'loocv'.")

    fold_accuracies = []

    # Perform cross-validation
    for _, (train_index, val_index) in enumerate(kf.split(X_train_full, y_train_full)):
        # Split data for the current fold
        X_train, X_val = X_train_full[train_index], X_train_full[val_index]
        y_train, y_val = y_train_full[train_index], y_train_full[val_index]
        
        # Scale data for the current fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=batch_size)

        # Build and train the model for the current fold
        activation_fn = getattr(nn, activation_name)
        model = DynamicDNN(input_size, hidden_size, num_classes, n_layers, activation_fn, dropout_rate)
        
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        trainer = PyTorchTrainer(model, nn.CrossEntropyLoss(), optimizer, device)
        
        best_fold_accuracy = trainer.train(train_loader, val_loader, num_epochs, patience=10, verbose=False)
        fold_accuracies.append(best_fold_accuracy)

    # Calculate the mean and standard deviation of accuracies across folds
    mean_accuracy = np.mean(fold_accuracies)
    std_dev = np.std(fold_accuracies)
    
    return mean_accuracy, std_dev

# --- Plotting Function ---
def plot_history(history):
    """Plots the training and validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Accuracy
    ax2.plot(history['train_accuracy'], label='Train Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

# --- Callback for printing progress ---
def print_progress(study, trial):
    """A callback function to print progress during Optuna study."""
    print(f"Trial {trial.number} finished.")
    print(f"  Value: (Acc: {trial.values[0]:.4f}, Std: {trial.values[1]:.4f})")
    if study.best_trials:
        # For multi-objective, best_trials[0] is one of the best, not necessarily *the* best
        best_trial = study.best_trials[0]
        print(f"  Current Pareto front size: {len(study.best_trials)}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    CV_METHOD = 'kfold'          # Options: 'kfold' or 'loocv'
    N_TRIALS = 50                # Number of Optuna trials to run
    NUM_EPOCHS_OPTUNA = 50       # Max epochs for each Optuna trial
    NUM_EPOCHS_FINAL = 200       # Max epochs for final training (will stop early)
    FINAL_MODEL_PATIENCE = 15    # Patience for final model's early stopping
    STABILITY_THRESHOLD = 0.05   # Max std dev for a trial to be considered "stable"

    # Device Selection
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Prepare Data
    X_train_full, y_train_full, X_test, y_test, class_names = prepare_data()
    input_size, num_classes = X_train_full.shape[1], len(set(y_train_full))
    
    # Run Optuna Hyperparameter Search
    print(f"\n--- Starting Optuna search with {N_TRIALS} trials (CV Method: {CV_METHOD}) ---")
    sampler = TPESampler(n_startup_trials=N_TRIALS // 2)
    study = optuna.create_study(directions=['maximize', 'minimize'], sampler=sampler)
    
    study.optimize(
        lambda trial: objective(trial, X_train_full, y_train_full, input_size, num_classes, NUM_EPOCHS_OPTUNA, DEVICE, cv_method=CV_METHOD), 
        n_trials=N_TRIALS, 
        callbacks=[print_progress]
    )

    # --- Select Best Trial and Train Final Model ---
    print("\n--- Optuna search complete ---")
    
    # Select the best trial using the two-step stability-then-accuracy method
    print(f"Found {len(study.best_trials)} optimal trials on the Pareto front.")
    stable_trials = [t for t in study.best_trials if t.values[1] < STABILITY_THRESHOLD]
    
    if not stable_trials:
        print(f"No trials met the stability threshold of {STABILITY_THRESHOLD}, selecting the highest accuracy trial overall.")
        best_trial = max(study.best_trials, key=lambda t: t.values[0])
    else:
        print(f"Found {len(stable_trials)} stable trials. Selecting the most accurate among them.")
        best_trial = max(stable_trials, key=lambda t: t.values[0])

    print(f"\nSelected Best Trial #{best_trial.number}")
    print(f"  Mean CV Accuracy: {best_trial.values[0]:.4f}")
    print(f"  Std Dev: {best_trial.values[1]:.4f}")
    print("  Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Prepare data and loaders for the final training run
    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_full)
    X_test_final = scaler.transform(X_test)
    
    best_batch_size = best_trial.params['batch_size']
    
    test_loader_final = DataLoader(TensorDataset(torch.tensor(X_test_final, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=best_batch_size)

    # Build the final model with the best hyperparameters
    final_model = DynamicDNN(
        input_size=input_size, num_classes=num_classes,
        hidden_size=best_trial.params['hidden_size'], num_layers=best_trial.params['n_layers'],
        activation_fn=getattr(nn, best_trial.params['activation']), dropout_rate=best_trial.params['dropout_rate']
    )
    final_optimizer = getattr(optim, best_trial.params['optimizer'])(final_model.parameters(), lr=best_trial.params['lr'])
    final_scheduler = torch.optim.lr_scheduler.StepLR(final_optimizer, step_size=25, gamma=0.1)
    
    # Instantiate the final trainer
    final_trainer = PyTorchTrainer(final_model, nn.CrossEntropyLoss(), final_optimizer, DEVICE)

    # Create a validation split for the final training run to monitor progress and for early stopping
    print(f"\n--- Training final model for up to {NUM_EPOCHS_FINAL} epochs (Patience: {FINAL_MODEL_PATIENCE}) ---")
    X_train_plot, X_val_plot, y_train_plot, y_val_plot = train_test_split(X_train_final, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)
    train_loader_plot = DataLoader(TensorDataset(torch.tensor(X_train_plot, dtype=torch.float32), torch.tensor(y_train_plot, dtype=torch.long)), batch_size=best_batch_size, shuffle=True)
    val_loader_plot = DataLoader(TensorDataset(torch.tensor(X_val_plot, dtype=torch.float32), torch.tensor(y_val_plot, dtype=torch.long)), batch_size=best_batch_size)
    
    # Train the final model
    final_trainer.train(train_loader_plot, val_loader_plot, NUM_EPOCHS_FINAL, patience=FINAL_MODEL_PATIENCE, scheduler=final_scheduler, verbose=True)
    
    # Evaluate the final model on the unseen test set
    final_trainer.evaluate(test_loader_final, class_names=class_names)
    
    # Plot the training history of the final model
    plot_history(final_trainer.history)