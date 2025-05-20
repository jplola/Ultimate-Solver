from torch import optim
import torch
from torch.utils.data import random_split, DataLoader



def train_model(model,tensordataset,lr,batch_size,num_epochs):
    """
    model: Neural Network (CNN Type)
    tensordataset = TensorDataset containing: model_input, target probabilities, target values
    lr: learning rate
    batch_size: batch_size
    num_epochs: number of epochs

    Returns:
        trained model
    """
    val_const = 1.
    prob_const = 1.
    l2_const = 1e-5


    device = torch.device("mps")


    # Split the dataset into training and test sets (e.g., 80% training, 20% testing)
    train_size = int(0.9 * len(tensordataset))
    test_size = len(tensordataset) - train_size
    print(f'len dataset = {len(tensordataset)}')
    train_dataset, test_dataset = random_split(tensordataset, [train_size, test_size])

    # Create DataLoaders for both training and testing sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss functions
    value_loss_fn = torch.nn.MSELoss()  # For scalar value output
    policy_loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    torch.autograd.set_detect_anomaly(True)

    model.to(device)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_train_value_loss = 0.0
        total_train_policy_loss = 0.0

        # Training phase
        for boards, target_probs, target_value in train_loader:
            boards = boards.to(device).float()
            target_probs = target_probs.to(device)
            target_value = target_value.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_value, pred_policy = model(boards)


            # Compute value loss
            value_loss = value_loss_fn(pred_value.view((pred_value.shape[0],)), target_value)

            # Compute policy loss
            policy_loss = policy_loss_fn(pred_policy, target_probs)

            # L2 regularization
            l2_reg = sum(param.pow(2).sum() for param in model.parameters()) * l2_const

            # Total loss
            loss = policy_loss + l2_reg + value_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate training loss for monitoring
            total_train_value_loss += value_loss.item()
            total_train_policy_loss += policy_loss.item()

        # Calculate average training losses
        avg_train_value_loss = total_train_value_loss / len(train_loader)
        avg_train_policy_loss = total_train_policy_loss / len(train_loader)

        # Testing phase
        model.eval()  # Set the model to evaluation mode
        total_test_value_loss = 0.0
        total_test_policy_loss = 0.0

        with torch.no_grad():  # No gradients needed for testing
            for boards, target_probs, target_value in test_loader:
                boards = boards.to(device)
                target_probs = target_probs.to(device)
                target_value = target_value.to(device)
                # Forward pass
                pred_value, pred_policy = model(boards)

                # Compute value loss
                value_loss = value_loss_fn(pred_value.view((pred_value.shape[0],)), target_value) * val_const

                # Compute policy loss
                policy_loss = policy_loss_fn(pred_policy, target_probs) * prob_const

                # Accumulate test loss for monitoring
                total_test_value_loss += value_loss.item()
                total_test_policy_loss += policy_loss.item()

        # Calculate average test losses
        avg_test_value_loss = total_test_value_loss / len(test_loader)
        avg_test_policy_loss = total_test_policy_loss / len(test_loader)

        # Print average losses per epoch for training and testing
        model.save_model(f'/Users/pietropezzoli/Desktop/Thesis Pietro Pezzoli/tesi/pythonProject/Ultimate-Solver/checkpoints/AlphaCheckpoints/model_checkpoints/mcts_{epoch}.pth')
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(
            f"  Training -> Value Loss: {avg_train_value_loss:.4f}, Policy Loss: {avg_train_policy_loss:.4f}, L2: {l2_reg:.4f}")
        print(f"  Testing  -> Value Loss: {avg_test_value_loss:.4f}, Policy Loss: {avg_test_policy_loss:.4f}")

    print("Training complete!")

    return model


def save_model_weights(model, path):
    """
    Save the model weights to a specified path.

    Args:
        model (PolicyValueNet): The model instance.
        path (str): The file path where to save the model weights.
    """
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def load_model_weights(path,model_class):
    """
    Load model weights from a specified path into a new PolicyValueNet instance.

    Args:
        path (str): The file path from where to load the model weights.

    Returns:
        PolicyValueNet: A model instance with loaded weights.
    """
    # Initialize a new model instance
    model = model_class()

    # Load the state dictionary
    model.load_state_dict(torch.load(path,weights_only=True))
    print(f"Model weights loaded from {path}")

    return model


