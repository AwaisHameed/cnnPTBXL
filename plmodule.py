import lightning.pytorch as pl
from models import * # to get the class name from the config
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import ast


class ECGModel(pl.LightningModule):
    
    # Init method
    def __init__(self,  
                optimizer_name = "Adam",
                optimizer_hparams = {"lr":0.0001},
                lr_scheduler_hparams = {"step_size": 1},
                prediction_threshold = 0.5, # Threshold used for prediction
                model_class = None,
                model_param = {"input_channels": 12, "output_dim":5, "hidden_dim": 32, "dropout": 0.2}
                ):
        
        super().__init__()
    
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        self.prediction_threshold = prediction_threshold

        self.loss_module = nn.BCELoss() # Loss module
         
         # Model is defined
        self.model = eval(model_class)(**model_param)
        
        # Lists storing metrics for each step (validation phase)
        self.validation_step_outputs_acc = []
        self.validation_step_outputs_precision = []
        self.validation_step_outputs_recall = []
        self.validation_step_outputs_f1 = []

        # Lists storing metrics for each step (testing phase)
        self.test_step_outputs_acc = []
        self.test_step_outputs_precision = []
        self.test_step_outputs_recall = []
        self.test_step_outputs_f1 = []

    # For every step during training
    def training_step(self, batch, batch_idx):
        ecg = batch["ecg"]
        gt = batch["class"]
        pred = self.model(ecg)
        loss = self.loss_module(pred, gt)

        acc = ((pred > self.prediction_threshold) == gt).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        pass


    def validation_step(self, batch, batch_idx):
        ecg = batch["ecg"]
        gt = batch["class"]
        pred = self.model(ecg)
        loss = self.loss_module(pred, gt)

        # Calculate accuracy
        acc = ((pred > self.prediction_threshold) == gt).float().mean()
        # Log loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        
        # Calculate precision, recall, and F1-score
        pred_labels = (pred > self.prediction_threshold).cpu().numpy()
        true_labels = gt.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro')
        
        #For each step add value to list
        self.validation_step_outputs_acc.append(acc)
        self.validation_step_outputs_precision.append(precision)
        self.validation_step_outputs_recall.append(recall)
        self.validation_step_outputs_f1.append(f1)

    def on_validation_epoch_end(self):
        # Log all metrics at end of validation
        all_acc = torch.stack(self.validation_step_outputs_acc)
        self.log("val_epoch_acc", all_acc.mean())
        all_precision = np.mean(self.validation_step_outputs_precision)
        self.log("val_epoch_precision", all_precision)
        all_recall = np.mean(self.validation_step_outputs_recall)
        self.log("val_epoch_recall", all_recall)
        all_f1 = np.mean(self.validation_step_outputs_f1)
        self.log("val_epoch_f1", all_f1)

        # Clear the lists
        self.validation_step_outputs_acc.clear()
        self.validation_step_outputs_precision.clear()
        self.validation_step_outputs_recall.clear()
        self.validation_step_outputs_f1.clear()
    
    # For every step during testing
    def test_step(self, batch, batch_idx):
        ecg = batch["ecg"]
        gt = batch["class"]
        pred = self.model(ecg)

        # Calculate accuracy
        acc = ((pred > self.prediction_threshold) == gt).float().mean()

        # Calculate precision, recall, and F1-score
        pred_labels = (pred > self.prediction_threshold).cpu().numpy()
        true_labels = gt.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
        
        #For each step add value to list
        self.test_step_outputs_precision.append(precision)
        self.test_step_outputs_recall.append(recall)
        self.test_step_outputs_f1.append(f1)
        
        # Calculate mean of values added to list
        all_precision = np.mean(self.test_step_outputs_precision)
        all_recall = np.mean(self.test_step_outputs_recall)
        all_f1 = np.mean(self.test_step_outputs_f1)

        # Log the metrics
        self.log("test_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("test_pres", all_precision, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("test_recall", all_recall, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        self.log("test_f1", all_f1, prog_bar=True, on_epoch=True, sync_dist=True, logger=True)
        
        # Calculate and show confusion matrix
        #cm = multilabel_confusion_matrix(true_labels, pred_labels)
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        #disp.plot()
        #plt.show()



    

    def configure_optimizers(self):

        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(), **self.hparams.optimizer_hparams)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.hparams.lr_scheduler_hparams)
        return [optimizer], [lr_scheduler]
