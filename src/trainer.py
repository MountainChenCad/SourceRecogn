# src/trainer.py
import torch
# import torch.nn as nn # Already imported in models.py
# import torch.optim as optim # Already imported in main.py
from tqdm import tqdm
import os
import numpy as np
from .utils import save_checkpoint, load_checkpoint, plot_training_curves # Import new plot function
import logging # Import logging

class Trainer:
    def __init__(self, model, criterion, optimizer, device, results_dir, save_name_prefix): # Renamed save_name to save_name_prefix
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.results_dir = results_dir
        self.save_name_prefix = save_name_prefix # Used as a prefix for model files
        self.best_val_acc = 0.0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}


    def train_epoch(self, data_loader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(data_loader, desc="Training", leave=False, ncols=100)
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
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%" if total > 0 else "0.00%")
        epoch_loss = total_loss / total if total > 0 else 0
        epoch_acc = 100. * correct / total if total > 0 else 0
        return epoch_loss, epoch_acc

    def evaluate(self, data_loader, desc="Evaluating", get_features_for_tsne=False):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_true_labels, all_predicted_labels, all_features_list = [], [], []
        progress_bar = tqdm(data_loader, desc=desc, leave=False, ncols=100)
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
                    if hasattr(self.model, 'get_features'):
                        features = self.model.get_features(inputs)
                        all_features_list.append(features.cpu().numpy())
                    elif not hasattr(self, '_warned_no_features_eval'):
                        logging.warning("Model does not have 'get_features'. t-SNE in eval cannot be generated.")
                        self._warned_no_features_eval = True
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%" if total > 0 else "0.00%")
        epoch_loss = total_loss / total if total > 0 else 0
        epoch_acc = 100. * correct / total if total > 0 else 0
        final_features = np.vstack(all_features_list) if get_features_for_tsne and all_features_list else np.array([])
        return epoch_loss, epoch_acc, np.array(all_true_labels), np.array(all_predicted_labels), final_features

    def train(self, train_loader, val_loader, epochs, visualizations_dir, plot_title_prefix=''):
        logging.info(f"Starting training for {epochs} epochs for '{self.save_name_prefix}'...")
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []} # Reset history

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_acc, _, _, _ = self.evaluate(val_loader, desc="Validating")
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                if val_acc > self.best_val_acc:
                    logging.info(f"Validation accuracy improved ({self.best_val_acc:.2f}% -> {val_acc:.2f}%). Saving model...")
                    self.best_val_acc = val_acc
                    save_path = os.path.join(self.results_dir, 'models', f'{self.save_name_prefix}_best.pth.tar')
                    save_checkpoint({'epoch': epoch + 1, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'best_val_acc': self.best_val_acc}, filename=save_path)
            else:
                logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                # Fill val history with NaN or skip if no validation
                self.history['val_loss'].append(float('nan'))
                self.history['val_acc'].append(float('nan'))
                if (epoch + 1) % 10 == 0 or epoch == epochs - 1: # Save periodically if no validation
                     save_path = os.path.join(self.results_dir, 'models', f'{self.save_name_prefix}_epoch_{epoch+1}.pth.tar')
                     save_checkpoint({'epoch': epoch + 1, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, filename=save_path)


        logging.info(f"Training finished for '{self.save_name_prefix}'.")
        final_save_path = os.path.join(self.results_dir, 'models', f'{self.save_name_prefix}_final.pth.tar')
        save_checkpoint({'epoch': epochs, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'best_val_acc': self.best_val_acc}, filename=final_save_path)
        logging.info(f"Final model saved to {final_save_path}")

        # Plot training curves
        plot_curves_save_path_prefix = os.path.join(visualizations_dir, self.save_name_prefix)
        plot_training_curves(self.history, plot_curves_save_path_prefix, title_prefix=f"{plot_title_prefix} ({self.save_name_prefix})")