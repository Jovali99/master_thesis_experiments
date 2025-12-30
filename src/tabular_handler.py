import numpy as np
import torch
from torch import cuda, device, optim, no_grad
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as T
import random
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "LeakPro")))

from leakpro import AbstractInputHandler
from leakpro.schemas import TrainingOutput, EvalOutput

class TabularInputHandler(AbstractInputHandler):
    """Class to handle the user input for the CIFAR structured datasets."""

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int | None = None,
        device = None
        ) -> TrainingOutput:
        """Model training procedure."""

        if epochs is None:
            raise ValueError("epochs not found in configs")

        # prepare training
        if device is None:
            device = torch.device("cuda" if cuda.is_available() else "cpu")
        model.to(device)

        accuracy_history = []
        loss_history = []

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            model.train()
            train_acc, train_loss, total_samples = 0.0, 0.0, 0

            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, position=1):
                #target = target.float().unsqueeze(1) # Used with BCE loss criterion and binary classification
                #data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                labels = labels.long()
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                pred = outputs.argmax(dim=1) 
                loss.backward()
                optimizer.step()

                #pred = output >= 0.5   # Binary classification

                # Accumulate performance of shadow model
                train_acc += pred.eq(labels.view_as(pred)).sum().item()
                total_samples += labels.size(0)
                train_loss += loss.item() * labels.size(0)

            avg_train_loss = train_loss / total_samples
            train_accuracy = train_acc / total_samples 

            accuracy_history.append(train_accuracy) 
            loss_history.append(avg_train_loss)

            print(f"Epoch {epoch+1} completed. Train Acc: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}")

        results = EvalOutput(accuracy = train_accuracy,
                             loss = avg_train_loss,
                             extra = {"accuracy_history": accuracy_history, "loss_history": loss_history})
        return TrainingOutput(model = model, metrics=results)

    def trainStudyFbD(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int|None = None,
        noise_std: float|None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ) -> TrainingOutput:
        """Model training procedure."""

        if epochs is None:
            raise ValueError("epochs not found in configs")

        # prepare training
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        accuracy_history = []
        loss_history = []
        
        # training loop
        for epoch in range(epochs):
            train_loss, train_acc, total_samples = 0, 0, 0
            model.train()
            for org_idx, batch_weights, inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                labels = labels.long()
                inputs, labels, batch_weights = inputs.to(gpu_or_cpu, non_blocking=True), labels.to(gpu_or_cpu, non_blocking=True), batch_weights.to(gpu_or_cpu, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                # Apply weighted loss
                weighted_loss = (loss * batch_weights).mean()

                weighted_loss.backward()

                # Add Gaussian noise to the gradients
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param.grad) * noise_std
                        param.grad += noise

                pred = outputs.argmax(dim=1) 
                optimizer.step()

                # Accumulate performance of shadow model
                train_acc += pred.eq(labels.view_as(pred)).sum().item()
                total_samples += labels.size(0)
                train_loss += weighted_loss.item() * labels.size(0)
                
            avg_train_loss = train_loss / total_samples
            train_accuracy = train_acc / total_samples 
            
            accuracy_history.append(train_accuracy) 
            loss_history.append(avg_train_loss)

            # Apply the step scheduler
            if scheduler is not None:
                scheduler.step()

            print(f"Epoch {epoch+1} completed. Train Acc: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}")

        results = EvalOutput(accuracy = train_accuracy,
                             loss = avg_train_loss,
                             extra = {"accuracy_history": accuracy_history, "loss_history": loss_history})
        return TrainingOutput(model = model, metrics=results)

    def eval(self, loader, model, criterion):
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()
        loss, acc = 0, 0
        total_samples = 0
        with no_grad():
            for batch in loader:
                # batch can be (org_idx, batch_weights, data, target) OR (data, target)
                if len(batch) == 4:
                    org_idx, batch_weights, data, target = batch
                elif len(batch) == 2:
                    data, target = batch
                else:
                    raise ValueError(f"Unexpected number of elements in batch: {len(batch)}")
                data, target = data.to(gpu_or_cpu), target.to(gpu_or_cpu)
                target = target.view(-1) 
                output = model(data)
                batch_loss = criterion(output, target).sum()
                loss += batch_loss.item()
                pred = output.argmax(dim=1) 
                acc += pred.eq(target).sum().item()
                total_samples += target.size(0)
            loss /= total_samples
            acc = float(acc) / total_samples
            
        output_dict = {"accuracy": acc, "loss": loss}
        return EvalOutput(**output_dict)

    class TabularUserDataset(AbstractInputHandler.UserDataset):
        def __init__(self, data, targets, normalize=True, one_hot=False, mean=None, std=None):
            """
            Args:
                data (Tensor): Tabular data of shape (N, D)
                targets (Tensor): Labels of shape (N,) or (N, num_classes) if one-hot
                normalize (bool): Whether to apply feature-wise normalization
                one_hot (bool): Whether to keep targets as one-hot vectors
                mean, std (Tensor, optional): Precomputed normalization stats
            """
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)

            assert data.shape[0] == targets.shape[0], "Data and targets must have the same length"
            assert data.dim() == 2, "Tabular data must be of shape (N, D)"

            self.data = data.float()  # Ensure float type
            self.normalize = normalize

            if one_hot:
                self.targets = targets.float()
            else:
                self.targets = targets.argmax(dim=1).long()

            if normalize:
                if mean is None or std is None:
                    self.mean = self.data.mean(dim=0)
                    self.std = self.data.std(dim=0).clamp(min=1e-8)
                else:
                    self.mean = mean
                    self.std = std

        def transform(self, x):
            """Normalize using stored mean and std."""
            if self.normalize:
                return (x - self.mean) / self.std 
            return x

        def __getitem__(self, index):
            x = self.data[index]
            y = self.targets[index]
            x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.targets)
