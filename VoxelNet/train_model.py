import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from voxelnet import VoxNet
from VoxelDataset import VDataset
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def train(train_loader, val_loader, num_epochs=10):
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

        # Validation (optional)
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

        val_losses.append(val_avg_loss)

    torch.save(model.state_dict(), 'model.pth')

    return train_losses, val_losses

if __name__ == '__main__':
    train_losses = []
    val_losses = []

    train_ds = VDataset(root_dir='Voxels')
    test_ds = VDataset(root_dir='Voxels', mode='test')
    print(len(train_ds))

    # Your model
    model = VoxNet(n_classes=train_ds.num_classes())  # change to your actual class count
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Data loaders (replace with your actual dataset)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Training loop
    num_epochs = 15

    train_losses, val_losses = train(train_loader, val_loader, num_epochs)

    plt.plot(train_losses, label="train losses", color='r')
    plt.plot(val_losses, label="valid losses", color='b')
    plt.legend()

    plt.show()

    model = VoxNet(n_classes=10)  # Initialize the model with the number of classes

    # Load the state dictionary (e.g., from a saved checkpoint)
    checkpoint = torch.load('model.pth')  # Load the model checkpoint
    model.load_state_dict(checkpoint)  # Load the state dict into the model

    model.eval()  # Switch to evaluation mode if you're testing the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_ds = VDataset(root_dir='Voxels', mode='test')
    val_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []

    # Iterate over the data
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Store the true labels and predicted labels
            all_preds.extend(preds.cpu().numpy())  # move to CPU for easier handling
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Plotting labels and ticks
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, list(test_ds.classes_encoding.keys()))
    plt.yticks(tick_marks, list(test_ds.classes_encoding.keys()))

    # Labeling axes
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Annotating each cell in the matrix with the numeric value
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.show()