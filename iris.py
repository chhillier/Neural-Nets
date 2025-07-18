import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# --- Model Definition ---
class SimpleDNN(nn.Module):
    """A simple feed-forward neural network."""
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
# --- Classifier Class ---
class IrisClassifier:
    """
    An all-in-one class to handle the training and evaluation of a DNN
    on the Iris dataset.
    """
    def __init__(self, input_size, hidden_size, num_classes, learning_rate, batch_size, num_epochs):
        """Initializes the classifier and its components."""
        # Store hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Set the device for training
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon (MPS) device.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using NVIDIA CUDA device.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device.")

        # Initialize model, loss function, and optimizer
        self.model = SimpleDNN(self.input_size, self.hidden_size, self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize history, data loaders, and scaler
        self.history = {'train_loss': [], 'validation_loss': [], 'train_accuracy': [], 'validation_accuracy': []}
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.scaler = StandardScaler()
        
        print(f"Model initialized: {self.model}")

    def load_and_prepare_data(self):
        """Loads and preprocesses the Iris dataset."""
        print("\n--- Loading and Preparing Iris Dataset ---")
        iris = load_iris()
        X, y = iris.data, iris.target
        self.feature_names = iris.feature_names
        self.class_names = iris.target_names

        # Create train/validation/test splits (60%/20%/20%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Create PyTorch DataLoaders
        train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print("Data preparation complete.")

    def train(self):
        """Runs the training and validation loop."""
        if not all([self.train_loader, self.val_loader]):
            raise RuntimeError("Data not loaded. Call load_and_prepare_data() first.")
        
        print("\n--- Training the model ---")
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for features, labels in self.val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item() * features.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Record and print metrics for the epoch
            train_epoch_loss = train_loss / train_total
            train_epoch_acc = train_correct / train_total
            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_correct / val_total
            
            self.history['train_loss'].append(train_epoch_loss)
            self.history['train_accuracy'].append(train_epoch_acc)
            self.history['validation_loss'].append(val_epoch_loss)
            self.history['validation_accuracy'].append(val_epoch_acc)

            print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {train_epoch_loss:.4f}, Acc: {train_epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")

    def evaluate(self):
        """Evaluates the model on the test set."""
        if not self.test_loader:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data() first.")
            
        print("\n--- Evaluating model on test set ---")
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in self.test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.class_names))

    def plot_history(self):
        """Plots the training and validation loss and accuracy."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['validation_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(); ax1.grid(True)
        
        ax2.plot(self.history['train_accuracy'], label='Train Accuracy')
        ax2.plot(self.history['validation_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend(); ax2.grid(True)
        
        plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define hyperparameters
    INPUT_SIZE = 4
    HIDDEN_SIZE = 8
    NUM_CLASSES = 3
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    NUM_EPOCHS = 1000

    # Create and run the classifier
    iris_classifier = IrisClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS
    )

    iris_classifier.load_and_prepare_data()
    iris_classifier.train()
    iris_classifier.evaluate()
    iris_classifier.plot_history()