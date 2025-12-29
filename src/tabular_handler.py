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

class TabularInputHandler(AbstractInputHandler)
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

            for data, target in dataloader:
                target = target.float().unsqueeze(1)
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)

                pred = output >= 0.5
                train_acc += pred.eq(target).sum().item()

                loss.backward()
                optimizer.step()
                train_loss += loss.item() 
                total_samples += target.size(0)

        train_acc = train_acc/len(dataloader.dataset)
        train_loss = train_loss/len(dataloader)


        output_dict = {"model": model, "metrics": {"accuracy": train_acc, "loss": train_loss}}
        output = TrainingOutput(**output_dict)

        return output

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

    class UserDataset(AbstractInputHandler.UserDataset):
        def __init__(self, data, targets, augment: bool = False, **kwargs):
            """
            Args:
                data (Tensor): Image data of shape (N, H, W, C) or (N, C, H, W)
                               Expected to be in range [0,1] (normalized).
                targets (Tensor): Corresponding labels.
                mean (Tensor, optional): Precomputed mean for normalization.
                std (Tensor, optional): Precomputed std for normalization.
            """
            assert data.shape[0] == targets.shape[0], "Data and targets must have the same length"
            assert data.max() <= 1.0 and data.min() >= 0.0, "Data should be in range [0,1]"

            self.data = data.float()  # Ensure float type
            self.targets = targets
            self.augment = augment

            for key, value in kwargs.items():
                setattr(self, key, value)
                
            if not hasattr(self, "mean") or not hasattr(self, "std"):
                # Reshape to (C, 1, 1) for broadcasting
                self.mean = self.data.mean(dim=(0, 2, 3)).view(-1, 1, 1)
                self.std = self.data.std(dim=(0, 2, 3)).view(-1, 1, 1)

            if not hasattr(self, "augment"):
                self.augment = False

            self.augment_transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(32, padding=4)
            ])

        def transform(self, x):
            """Normalize using stored mean and std."""
            return (x - self.mean) / self.std 

        def __getitem__(self, index):
            x = self.data[index]
            y = self.targets[index]

            # Horizontal flip
            if self.augment:
                x = self.augment_transforms(x)

            x = self.transform(x)

            return x, y

        def __len__(self):
            return len(self.targets)

        def __setstate__(self, state):
            self.__dict__.update(state)
            if not hasattr(self, "augment"):
                self.augment = False
            if not hasattr(self, "augment_transforms"):
                self.augment_transforms = T.Compose([
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomCrop(32, padding=4)
                ])

        def set_augment(self, augment: bool):
            self.augment = augment
