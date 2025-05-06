import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # Progress bar
import os
from .utils import save_checkpoint, load_checkpoint

class Trainer:
    def __init__(self, model, criterion, optimizer, device, results_dir, save_name):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.results_dir = results_dir
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

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)

        epoch_loss = total_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def evaluate(self, data_loader, desc="Evaluating"):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
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
                progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)

        epoch_loss = total_loss / total if total > 0 else 0
        epoch_acc = 100. * correct / total if total > 0 else 0
        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, epochs):
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)

            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader, desc="Validating")
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                 # Save checkpoint if validation accuracy improves
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
                 print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                 # Save model periodically or at the end if no validation set
                 if (epoch + 1) % 10 == 0 or epoch == epochs - 1: # Save every 10 epochs or last epoch
                     save_path = os.path.join(self.results_dir, 'models', f'{self.save_name}_epoch_{epoch+1}.pth.tar')
                     save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, filename=save_path)

        print("Training finished.")
        # Optionally save the final model state
        final_save_path = os.path.join(self.results_dir, 'models', f'{self.save_name}_final.pth.tar')
        save_checkpoint({
            'epoch': epochs,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc, # Include best val acc achieved
        }, filename=final_save_path)
        print(f"Final model saved to {final_save_path}")
