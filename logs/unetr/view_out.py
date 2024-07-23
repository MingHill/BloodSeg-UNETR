import matplotlib.pyplot as plt

import torch

from torch import nn 

from src.plotters import visualize_predictions
from src.utils import select_device
from src.unetr_trainer import UNETR_TRAINER
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_loss(df_train): 
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    plt.plot(df_train.index, df_train['Train Loss'], label='Train Loss')
    plt.plot(df_train.index, df_train['Valid Loss'], label='Valid Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def report_metrics(model, test_dataloader): 
    labels = {
        0: "background",
        1: "white blood cell",
        2: "platelet",
        3: "outer rbc",
        4: "reticulocyte (missing)",
        5: "inner rbc",
        6: "beads",
        7: "monster bead",
        8: "sensor scratch",
        9: "chambertop scratch",
        10: "debris",
        11: "bubble",
    }
    device = select_device()
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    trainer = UNETR_TRAINER(model=model, criterion=criterion, device = device)
    test_loss, class_report, gt, pred = trainer.test(data_batches=test_dataloader)
    print(f"Test Loss : {test_loss}")
    print(labels)
    print("\n")
    print(f" Class Report : \n {class_report}")
    print("\n")
    c_matrix = confusion_matrix(y_true=gt, y_pred=pred)
    print(c_matrix)

    
def segment(model, test_dataloader):
    device = select_device()
    model.eval()
    with torch.inference_mode():
        images, labels = next(iter(test_dataloader))
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        
        # Convert to numpy arrays for visualization
        images_np = images.cpu().numpy()
        labels_np = labels.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        print(f"Shape of image_np : {images_np.shape} | Label_np : {labels_np.shape} | Predictions : {predictions_np.shape}")
        # Visualize the predictions
        visualize_predictions(20, images=images_np, labels=labels_np, predictions=predictions_np)