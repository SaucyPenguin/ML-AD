import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchmetrics.classification import BinaryAUROC
from sklearn.ensemble import RandomForestClassifier
import pickle
import random
torch.manual_seed(42)

print("everything is imported")


def feature_selection(x, y, top_n):
    """
    Selects the top N features based on importance scores using a Random Forest classifier.

    Parameters:
        x (ndarray): Feature matrix.
        y (ndarray): Target labels.
        top_n (int): Number of top features to select.

    Returns:
        ndarray: Reduced feature matrix with only top N features.
    """
    print("Training Random Forest for feature selection...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # Creates RF classifier
    rf.fit(x, y)  # Trains and ranks SNPs on identification
    most_important = rf.feature_importances_  # Numpy array of feature importance scores
    top_indices = np.argsort(most_important)[-top_n:]  # Selects most important SNPs and returns indices
    top_feature_scores = most_important.tolist()
    with open("top_feature_scores.pkl", "wb") as f:
        pickle.dump(top_feature_scores, f)
    x_selected = x[:, top_indices]  # matrix, select all rows (:) but only top_indices columns
    with open("top_indices.pkl", "wb") as f1:
        pickle.dump(top_indices, f1)
    return x_selected


def create_npys(feature_dict):
    """
    Converts a feature dictionary into separate NumPy arrays for data and labels.

    Parameters:
        feature_dict (dict): Dictionary mapping sample IDs to (label, features).

    Returns:
        tuple: A tuple (X, y) where X is the feature matrix and y are the labels.
    """
    x = np.array(list(feature_dict.keys()), dtype=np.float32)
    y = np.array(list(feature_dict.values()), dtype=np.float32)
    # print("Feature matrix shape:", x.shape)  # (num_samples, num_features)
    # print("Labels shape:", y.shape) # (num_samples,)
    np.save("x.npy", x)
    np.save("y.npy", y)
    return x, y


def create_tensors(x, y):
    """
    Converts NumPy arrays to PyTorch tensors.

    Parameters:
        x (ndarray): Feature matrix.
        y (ndarray): Labels.

    Returns:
        tuple: A tuple (x_tensor, y_tensor) of PyTorch tensors.
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return x_tensor, y_tensor


def set_params(x_tensor, y_tensor):
    """
    Prepares training and testing data loaders from tensors.

    Parameters:
        x_tensor (Tensor): Feature tensor.
        y_tensor (Tensor): Label tensor.

    Returns:
        tuple: train_loader and test_loader for model training.
    """
    dataset = TensorDataset(x_tensor, y_tensor)
    num_samples = len(dataset)
    indices = list(range(num_samples))
    random.shuffle(indices)

    train_size = int(0.8 * len(dataset))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)
    return train_loader, test_loader


def create_model(x):
    """
    Initializes a neural network model based on the input feature size.

    Parameters:
        x (ndarray): Input data to determine feature dimension.

    Returns:
        nn.Module: A PyTorch neural network model.
    """
    input_dim = x.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 512),  # Input dim into 512 neurons (1st hidden)
        nn.ReLU(),  # Activation func of first hidden
        nn.Dropout(0.3),  # Randomly sets 30% of neurons to 0 to avoid over fitting
        nn.Linear(512, 256),  # Second hidden layer, size 256
        nn.ReLU(),  # Activation func of second hidden
        nn.Dropout(0.3),  # Same dropout percentage
        nn.Linear(256, 128),  # Third hidden layer, size 128
        nn.ReLU(),  # Activation func of third hidden
        nn.Dropout(0.3),  # Same dropout percentage
        nn.Linear(128, 1),  # Output layer, size 1
        nn.Sigmoid(),  # Activation func of output, between 0-1 for use with BC Entropy Loss
    )
    return model


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    """
    Saves the model and optimizer state to a checkpoint file.

    Parameters:
        model (nn.Module): The trained model.
        optimizer (Optimizer): The optimizer instance.
        epoch (int): The epoch number to record.
        path (str): File path to save the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    """
    Loads model and optimizer state from a checkpoint file.

    Parameters:
        model (nn.Module): The model to load weights into.
        optimizer (Optimizer): The optimizer to load state into.
        path (str): File path of the checkpoint.

    Returns:
        tuple: The model, optimizer, and epoch number.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]


def run_net(model, epochs, lr, train_loader, test_loader):
    """
    Trains and evaluates the neural network model.

    Parameters:
        model (nn.Module): The neural network to train.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        train_loader (DataLoader): Training data.
        test_loader (DataLoader): Testing data.

    Returns:
        float: Final accuracy on the test set.
    """
    criterion = nn.BCELoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr)  # Optimization function
    auc_metric = BinaryAUROC().to("cpu")  # Performance metric
    best_auc = 0
    epochs_wo_improvement = 0
    patience = 20
    for epoch in range(epochs):
        model.train()  # Model is initialized into training mode
        running_loss = 0.0  # Counter for running loss of epoch
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Clears out previous gradients to prevent accumulation
            outputs = model(inputs)  # Passes batch through model
            loss = criterion(outputs, labels)  # Evaluates loss
            loss.backward()  # Evaluates loss gradients with respect to model params
            optimizer.step()  # Updates model weights and biases
            running_loss += loss.item()  # Update running loss
        model.eval()  # Model is initialized to testing mode
        test_preds = []  # List initialized for predictions
        test_labels = []  # List initialized for actual labels
        with torch.no_grad():  # Turns off gradient tracking for better efficiency and prevent back prop
            for inputs, labels in test_loader:
                outputs = model(inputs)
                test_preds.append(outputs)
                test_labels.append(labels)
        test_preds = torch.cat(test_preds, dim=0)  # Combines all prediction batches into single tensors
        test_labels = torch.cat(test_labels, dim=0)  # Combines all label batches into single tensors
        test_auc = auc_metric(test_preds, test_labels).item()  # Evaluates AUC and Converts from tensor to float
        auc_metric.reset()
        if test_auc > best_auc:
            best_auc = test_auc
            epochs_wo_improvement = 0
        else:
            epochs_wo_improvement += 1

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Val AUC: {test_auc:.4f}")
        if epochs_wo_improvement >= patience:
            print(f"{epoch + 1} Epochs run. Best AUC: {best_auc:.4f}")
            return best_auc
    return best_auc


# with open("feature_dict.pkl", "rb") as F:
#     FEATURE_DICT = pickle.load(F)
# with open("snp_dict.pkl", "rb") as F1:
#     SNP_LIST = pickle.load(F1)
#
# X, Y = create_npys(FEATURE_DICT)
# X_SELECTED = feature_selection(X, Y, 1000)
# X_TENSOR, Y_TENSOR = create_tensors(X_SELECTED, Y)
# TRAIN_LOADER, TEST_LOADER = set_params(X_TENSOR, Y_TENSOR)
# MODEL = create_model(X_SELECTED)
# run_net(MODEL, 1000, 0.0001, TRAIN_LOADER, TEST_LOADER)
