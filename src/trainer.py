# src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np  # Add numpy
from .utils import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, model, criterion, optimizer, device, results_dir, save_name):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.results_dir = results_dir  # Main results directory for models
        self.save_name = save_name
        self.best_val_acc = 0.0

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(data_loader, desc="Training", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item(), acc=100. * correct / total if total > 0 else 0.0)

        epoch_loss = total_loss / total if total > 0 else 0
        epoch_acc = 100. * correct / total if total > 0 else 0
        return epoch_loss, epoch_acc

    def evaluate(self, data_loader, desc="Evaluating", get_features_for_tsne=False):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_true_labels = []
        all_predicted_labels = []
        all_features_list = []  # To store feature embeddings

        progress_bar = tqdm(data_loader, desc=desc, leave=False)

        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_true_labels.extend(labels.cpu().numpy())
                all_predicted_labels.extend(predicted.cpu().numpy())

                if get_features_for_tsne:
                    # Ensure your model has a get_features method
                    if hasattr(self.model, 'get_features'):
                        features = self.model.get_features(inputs)
                        all_features_list.append(features.cpu().numpy())
                    else:
                        # Fallback or warning if get_features is not available
                        if not hasattr(self, '_warned_no_features'):  # Warn only once
                            print("Warning: Model does not have a 'get_features' method. t-SNE cannot be generated.")
                            self._warned_no_features = True

                progress_bar.set_postfix(loss=loss.item(), acc=100. * correct / total if total > 0 else 0.0)

        epoch_loss = total_loss / total if total > 0 else 0
        epoch_acc = 100. * correct / total if total > 0 else 0

        final_features = np.vstack(all_features_list) if get_features_for_tsne and all_features_list else np.array([])

        return epoch_loss, epoch_acc, np.array(all_true_labels), np.array(all_predicted_labels), final_features

    def train(self, train_loader, val_loader, epochs, visualizations_dir=None,
              num_classes_for_plot=None):  # Added params
        print(f"Starting training for {epochs} epochs...")
        class_names_for_plot = [f"Class {i}" for i in range(num_classes_for_plot)] if num_classes_for_plot else None

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)

            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                # For validation, we don't typically generate full CM/t-SNE, just accuracy
                val_loss, val_acc, _, _, _ = self.evaluate(val_loader, desc="Validating", get_features_for_tsne=False)
                print(
                    f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                if val_acc > self.best_val_acc:
                    print(f"Validation accuracy improved ({self.best_val_acc:.2f}% -> {val_acc:.2f}%). Saving model...")
                    self.best_val_acc = val_acc
                    save_path = os.path.join(self.results_dir, 'models', f'{self.save_name}_best.pth.tar')
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_val_acc': self.best_val_acc,
                    }, filename=save_path)
            else:
                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    save_path = os.path.join(self.results_dir, 'models', f'{self.save_name}_epoch_{epoch + 1}.pth.tar')
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, filename=save_path)

        print("Training finished.")
        final_save_path = os.path.join(self.results_dir, 'models', f'{self.save_name}_final.pth.tar')
        save_checkpoint({
            'epoch': epochs,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, filename=final_save_path)
        print(f"Final model saved to {final_save_path}")