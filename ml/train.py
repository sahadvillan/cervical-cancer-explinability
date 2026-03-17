import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import get_dataloaders
from model import CervicalCancerModel

# Configuration
DATA_DIR = '../data/cervical_cancer_classification'
EPOCHS = 4  # Keep small for template/testing
BATCH_SIZE = 32
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    # Using 'weighted' average because classes might be imbalanced
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, acc, prec, rec, f1

def main():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader, test_loader, classes = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    print(f"Classes: {classes}")
    
    model = CervicalCancerModel(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    mlflow.set_experiment("Cervical_Cancer_Classification")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("architecture", "ResNet18")
        
        best_val_f1 = 0.0
        
        for epoch in range(EPOCHS):
            print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)
            
            # Save the model if it has the best F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), "best_model.pth")
                print("Saved new best model.")
                
        # Final test evaluation
        print("Running evaluation on test set...")
        # Load best model for testing
        if os.path.exists("best_model.pth"):
            model.load_state_dict(torch.load("best_model.pth"))
            
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion)
        print(f"Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")
        
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_prec", test_prec)
        mlflow.log_metric("test_rec", test_rec)
        mlflow.log_metric("test_f1", test_f1)
        
        # Log the PyTorch model artifact
        mlflow.pytorch.log_model(model, "model")
        
        # Save ML metadata for the Chatbot to consume
        metadata = {
            "dataset": "Cervical Cancer Classification",
            "classes": classes,
            "architecture": "ResNet18",
            "test_accuracy": round(test_acc, 4),
            "test_f1": round(test_f1, 4),
            "test_precision": round(test_prec, 4),
            "test_recall": round(test_rec, 4),
            "training_epochs": EPOCHS,
            "training_samples": len(train_loader.dataset)
        }
        
        with open("model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
            
        mlflow.log_artifact("model_metadata.json")

if __name__ == "__main__":
    main()
