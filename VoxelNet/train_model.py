import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from voxelnet import VoxNet
from VoxelDataset import VDataset, VoxelAugmentation
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.nn.functional.log_softmax(pred, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = torch.nn.functional.nll_loss(log_preds, target)
        return (1 - self.smoothing) * nll + self.smoothing * loss / n_classes
    
def train(train_loader, val_loader, num_epochs=10):
    best_val_accuracy = 0.0
    drop_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for voxels, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):            
            voxels, labels = voxels.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(voxels)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * voxels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = train_loss / total
        acc = correct / total
        print(f"Train Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
        train_losses.append(avg_loss)

        # Validation
        val_acc, val_avg_loss = validate(val_loader)
        val_losses.append(val_avg_loss)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
            }, 'model.pth')
            drop_counter = 0
        else:
            drop_counter += 1
            if drop_counter > 4:
                print('Not improving. Stopping training')
                break

        scheduler.step(val_avg_loss)

    return train_losses, val_losses

def validate(val_loader):
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for voxels, labels in val_loader:
            voxels, labels = voxels.to(device), labels.to(device)
            outputs = model(voxels)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * voxels.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_avg_loss = val_loss / val_total
    val_acc = val_correct / val_total
    print(f"Val Loss: {val_avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")
    
    return val_acc, val_avg_loss

if __name__ == '__main__':
    train_losses = []
    val_losses = []

    train_ds = VDataset(
    root_dir='../Voxels',
    transform=VoxelAugmentation()
    )
    test_ds = VDataset(root_dir='../Voxels', mode='test')
    print(len(train_ds))

    # Your model
    model = VoxNet(n_classes=train_ds.num_classes())  # change to your actual class count
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss and optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # Enable CPU optimizations
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    train_loader = DataLoader(
        train_ds, 
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True
    )

    val_loader = DataLoader(
        test_ds, 
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True
    )


    # Training loop
    num_epochs = 15

    train_losses, val_losses = train(train_loader, val_loader, num_epochs)

    plt.plot(train_losses, label="train losses", color='r')
    plt.plot(val_losses, label="valid losses", color='b')
    plt.legend()

    plt.show()

    # Load and evaluate the model
    model = VoxNet(n_classes=train_ds.num_classes())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    checkpoint = torch.load('model.pth', map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from epoch {checkpoint['epoch']} with best val accuracy: {checkpoint['best_val_accuracy']:.4f}")

    model.eval()

    test_ds = VDataset(root_dir='../Voxels', mode='test')
    val_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, list(test_ds.classes_encoding.keys()), rotation=45)
    plt.yticks(tick_marks, list(test_ds.classes_encoding.keys()))

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], 
                    horizontalalignment="center", 
                    verticalalignment="center", 
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.show()