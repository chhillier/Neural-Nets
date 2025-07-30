import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import copy
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import json
import os
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from mymodelzoo.cnn import DynamicCNN

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
    params = {
        'activation_name': trial.suggest_categorical('activation_name', ['ReLU', 'GELU', 'SiLU']),
        'n_conv_blocks': trial.suggest_int('n_conv_blocks', 2, 4),
        'base_channels': trial.suggest_int('base_channels', 16, 48, log=True),
        'block_conv_layers': trial.suggest_int('block_conv_layers', 1, 2),
        'kernel_size': trial.suggest_categorical('kernel_size', [3, 5]),
        'n_fc_layers': trial.suggest_int('n_fc_layers', 1, 2),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'fc_size': trial.suggest_int('fc_size', 64, 256, log=True),
        'optimizer_name': trial.suggest_categorical('optimizer_name', ['AdamW', 'Adam']),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
        'scheduler_name': trial.suggest_categorical('scheduler_name', ['StepLR', 'CosineAnnealingLR', 'OneCycleLR'])
    }
    if params['scheduler_name'] == 'StepLR':
        params['scheduler_step_size'] = trial.suggest_int('scheduler_step_size', 5, 15)
        params['scheduler_gamma'] = trial.suggest_float('scheduler_gamma', 0.1, 0.9)
    return params

class PyTorchTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    def train_epoch(self, data_loader):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        for features, labels in data_loader:
            features, labels = features.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
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
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss / total, correct / total

    def train(self, train_loader, val_loader, num_epochs, patience, verbose=False):
        best_val_loss, patience_counter, best_model_state = float('inf'), 0, None
        best_val_acc, train_acc_at_best = 0.0, 0.0

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)
            
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
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
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        print("\n--- Final Model Evaluation on Test Set ---")
        print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

def prepare_data():
    train_val_dataset = FashionMNIST(root='./data', train=True, download=True, transform=None)
    test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=None)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return train_val_dataset, test_dataset, class_names

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['train_loss'], label='Train Loss'); ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_accuracy'], label='Train Accuracy'); ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy'); ax2.legend(); ax2.grid(True)
    plt.show()

def objective(trial, train_dataset, val_dataset, input_shape, num_classes, num_epochs, device):
    params = define_hyperparameters(trial)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], num_workers=2)
    
    model = DynamicCNN(params, input_shape, num_classes)
    optimizer = getattr(optim, params['optimizer_name'])(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    if params['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['scheduler_step_size'], gamma=params['scheduler_gamma'])
    elif params['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif params['scheduler_name'] == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params['lr'], epochs=num_epochs, steps_per_epoch=len(train_loader))
    
    trainer = PyTorchTrainer(model, nn.CrossEntropyLoss(), optimizer, scheduler, device)
    val_accuracy, train_accuracy = trainer.train(train_loader, val_loader, num_epochs, patience=7, verbose=False)
    generalization_gap = abs(train_accuracy - val_accuracy)
    
    return val_accuracy, generalization_gap

def print_progress(study, trial):
    print(f"Trial {trial.number} finished.")
    if trial.state == optuna.trial.TrialState.COMPLETE:
        acc, gap = trial.values
        print(f"  Values: (Acc: {acc:.4f}, Gap: {gap:.4f})")
    if study.best_trials:
        print(f"  Current Pareto front size: {len(study.best_trials)}")

if __name__ == "__main__":
    LOAD_PARAMS_FROM_FILE = False
    PARAMS_PREFIX = "outputs/cnn_split_best_hyperparameters"
    MODEL_PREFIX = "models/cnn_split_best_model"
    LATEST_SUFFIX = "_latest"
    
    N_TRIALS = 50
    NUM_EPOCHS_OPTUNA = 30
    NUM_EPOCHS_FINAL = 50
    FINAL_MODEL_PATIENCE = 10
    GENERALIZATION_GAP_THRESHOLD = 0.10

    set_seed(72125)
    os.makedirs(os.path.dirname(PARAMS_PREFIX), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PREFIX), exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE} 🚀")

    full_train_dataset, test_dataset, class_names = prepare_data()
    
    print("Calculating dataset statistics...")
    train_data_tensors = torch.stack([transforms.ToTensor()(img) for img, _ in full_train_dataset])
    mean, std = train_data_tensors.mean(), train_data_tensors.std()
    
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    full_train_dataset.transform = data_transform
    test_dataset.transform = data_transform
    input_shape, num_classes = (1, 28, 28), len(class_names)
    
    train_indices, val_indices = train_test_split(
        list(range(len(full_train_dataset))),
        test_size=0.15,
        stratify=full_train_dataset.targets,
        random_state=42
    )
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)
    
    params_file_to_load = f"{PARAMS_PREFIX}{LATEST_SUFFIX}.json"

    if LOAD_PARAMS_FROM_FILE and os.path.exists(params_file_to_load):
        with open(params_file_to_load, 'r') as f:
            best_params = json.load(f)
    else:
        print(f"\n--- Starting Optuna search with {N_TRIALS} trials ---")
        sampler = TPESampler(n_startup_trials=N_TRIALS // 2, seed=72125)
        study = optuna.create_study(directions=['maximize', 'minimize'], sampler=sampler)
        
        study.optimize(
            lambda trial: objective(trial, train_subset, val_subset, input_shape, num_classes, NUM_EPOCHS_OPTUNA, DEVICE),
            n_trials=N_TRIALS,
            callbacks=[print_progress]
        )

        print("\n--- Optuna search complete ---")
        if OPTUNA_VIZ_INSTALLED:
            print("\n--- Generating Optuna Visualizations ---")
            plot_pareto_front(study, target_names=["Val Acc", "Train-Val Gap"]).show()
            plot_param_importances(study, target=lambda t: t.values[0], target_name="Validation Accuracy").show()
        else:
            print("\nSkipping visualizations: `plotly` is not installed.")
        
        stable_trials = [t for t in study.best_trials if t.values[1] < GENERALIZATION_GAP_THRESHOLD]
        if not stable_trials: stable_trials = study.best_trials
        best_trial = max(stable_trials, key=lambda t: t.values[0])
        best_params = best_trial.params
        
        acc, gap = best_trial.values
        results_to_save = {
            "objectives": {"validation_accuracy": acc, "generalization_gap": gap},
            "hyperparameters": best_params
        }
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"{PARAMS_PREFIX}_{timestamp}.json", 'w') as f: json.dump(results_to_save, f, indent=4)
        with open(params_file_to_load, 'w') as f: json.dump(best_params, f, indent=4)
        print(f"\nParameters for best run saved to {params_file_to_load}")
    
    print("\n--- Training final model with best discovered architecture ---")
    final_model = DynamicCNN(best_params, input_shape, num_classes)
    
    final_train_loader = DataLoader(train_subset, batch_size=best_params['batch_size'], shuffle=True)
    final_val_loader = DataLoader(val_subset, batch_size=best_params['batch_size'])
    final_test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

    final_optimizer = getattr(optim, best_params['optimizer_name'])(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    
    if best_params['scheduler_name'] == 'StepLR':
        final_scheduler = optim.lr_scheduler.StepLR(final_optimizer, step_size=best_params['scheduler_step_size'], gamma=best_params['scheduler_gamma'])
    elif best_params['scheduler_name'] == 'CosineAnnealingLR':
        final_scheduler = optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=NUM_EPOCHS_FINAL)
    elif best_params['scheduler_name'] == 'OneCycleLR':
        final_scheduler = optim.lr_scheduler.OneCycleLR(final_optimizer, max_lr=best_params['lr'], epochs=NUM_EPOCHS_FINAL, steps_per_epoch=len(final_train_loader))

    final_trainer = PyTorchTrainer(final_model, nn.CrossEntropyLoss(), final_optimizer, final_scheduler, DEVICE)
    final_trainer.train(final_train_loader, final_val_loader, NUM_EPOCHS_FINAL, patience=FINAL_MODEL_PATIENCE, verbose=True)
    
    plot_history(final_trainer.history)
    final_trainer.evaluate(final_test_loader, class_names=class_names)
    
    if 'timestamp' not in locals(): timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(final_trainer.model.state_dict(), f"{MODEL_PREFIX}_{timestamp}.pth")
    torch.save(final_trainer.model.state_dict(), f"{MODEL_PREFIX}{LATEST_SUFFIX}.pth")
    print(f"\n--- Final model saved to {MODEL_PREFIX}{LATEST_SUFFIX}.pth ---")