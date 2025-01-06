import os
import logging

import numpy as np
import pandas as pd

import torch
from sklearn.metrics import classification_report


class UNETR_TRAINER:
    def __init__(
        self,
        model,
        criterion,
        optimizer=None,
        scheduler=None,
        log_dir="./logs",
        device="cpu",
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_dir = log_dir
        self.device = device
        self.criterion = criterion
        self.model = model

    def fit(
        self,
        num_epochs,
        train_batches,
        valid_batches,
        train_eval_batches,
        freeze_threshold=None,
        save_checkpoint=False,
        log_batch_loss=False,
    ):
        self.model.train()

        train_losses, valid_losses = [], []
        for epoch in range(num_epochs):
            # if epoch == freeze_threshold:
            #     for param in self.model.encoder.parameters():
            #         param.requires_grad = True
            #     self.optimizer.add_param_group({'params': self.model.encoder.parameters()})

            for i, (input, targets) in enumerate(train_batches):
                input, targets = input.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(input)

                output_loss = self.criterion(outputs, targets)
                output_loss.backward()
                self.optimizer.step()

                if log_batch_loss:
                    logging.info(
                        f"Epoch {epoch + 1}/{num_epochs}, "
                        f"Batch {i + 1}/{len(train_batches)} "
                        f"Batch Loss: {output_loss.item():.4f}"
                    )

            val_loss = self.eval(valid_batches)
            train_loss = self.eval(train_eval_batches)

            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, "
            )

            valid_losses.append(val_loss)
            train_losses.append(train_loss)
        df = pd.DataFrame(
            {
                "Epoch": range(1, len(train_losses) + 1),
                "Train Loss": train_losses,
                "Valid Loss": valid_losses,
            }
        )
        df.to_csv(os.path.join(self.log_dir, "training_results.csv"), index=False)

        if save_checkpoint:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                os.path.join(self.log_dir, "model.pth"),
            )

        return train_losses, valid_losses

    def eval(self, data_batches):
        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        for i, (inputs, targets) in enumerate(data_batches):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(dim=0)
            num_samples += inputs.size(dim=0)
        self.model.train()
        loss = total_loss / num_samples
        return loss

    def test(self, data_batches):
        self.model.eval()

        total_loss = 0.0
        num_samples = 0

        total_ground_truth = []
        total_predicted = []
        for i, (inputs, targets) in enumerate(data_batches):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(dim=0)
            num_samples += inputs.size(dim=0)

            # To calculate metrics, similar to ignore index 255
            mask = targets != 255

            # permute to bring the class dimension last and then apply mask
            masked_outputs = outputs.permute(0, 2, 3, 1)[mask]
            masked_targets = targets[mask]

            predicted = torch.argmax(masked_outputs, dim=1)
            total_predicted.extend(predicted.cpu().numpy())
            total_ground_truth.extend(masked_targets.cpu().numpy())

        total_ground_truth = np.array(total_ground_truth)
        total_predicted = np.array(total_predicted)

        class_report = classification_report(
            y_true=total_ground_truth, y_pred=total_predicted, zero_division=0
        )

        self.model.train()
        loss = total_loss / num_samples

        return loss, class_report, total_ground_truth, total_predicted
