import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold

class MyModel(nn.Module):
    # Define your model architecture here

class MyDataset(Dataset):
    # Define your dataset here

def k_fold_cross_validation(k, dataset, model_class, loss_fn, optimizer_class, device, train_transform, test_transform):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for train_index, val_index in kf.split(dataset):
        # Create training and validation datasets for the current fold
        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)

        # Update the transformations for the training and validation subsets
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = test_transform

        # Create data loaders for the training and validation subsets
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Train the model on the training data
        model = model_class().to(device)
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            model.train()
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

        # Evaluate the model on the validation data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        fold_accuracy = correct / total
        accuracies.append(fold_accuracy)

    return accuracies

#########################################################################
num_epochs = 20
batch_size = 64
learning_rate = 0.001
dataset = MyDataset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss()  # Or any other appropriate loss function

results = k_fold_cross_validation(
    k=5,
    dataset=dataset,
    model_class=MyModel,
    loss_fn=loss_fn,
    optimizer_class=optim.Adam,
    device=device
)

print(f"Average Validation Loss: {sum(results) / len(results):.4f}")
