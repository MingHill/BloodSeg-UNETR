import os
import logging

import pandas as pd

import torch


class Trainer:
    def __init__(
        self, model, optimizer, scheduler=None, log_dir="./logs", device="cpu"
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.log_dir = log_dir

    def fit(
        self,
        num_epochs,
        train_batches,
        valid_batches,
        train_eval_batches,
        num_eval_batches=None,
        log_batch_loss=False,
        save_checkpoint=False,
    ):
        # set model to training mode
        self.model.train()

        train_losses, valid_losses = [], []
        for epoch in range(num_epochs):

            for i, X in enumerate(train_batches):
                inputs = X.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(pixel_values=inputs)

                batch_loss = outputs.loss  # using built-in loss function from HF
                batch_loss.backward()

                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                if log_batch_loss:
                    logging.info(
                        f"Epoch {epoch + 1}/{num_epochs}, "
                        f"Batch {i + 1}/{len(train_batches)} "
                        f"Batch Loss: {batch_loss.item():.4f}"
                    )

            train_loss = self.evaluate(train_eval_batches, num_eval_batches)
            valid_loss = self.evaluate(valid_batches, num_eval_batches)

            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, "
            )

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

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

    def evaluate(self, dataloader, num_eval_batches=None):
        # set model to evaluation mode
        self.model.eval()

        num_samples = 0
        total_loss = 0.0

        with torch.no_grad():
            for i, X in enumerate(dataloader):
                inputs = X.to(self.device)

                num_samples += inputs.size(dim=0)

                outputs = self.model(pixel_values=inputs)

                loss = outputs.loss  # using built-in loss function from HF
                total_loss += loss.item() * inputs.size(dim=0)

                if i == num_eval_batches:
                    break

        # set model back to training mode
        self.model.train()

        loss = total_loss / num_samples

        return loss
