import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src import data_setup, engine, model_builder, utils

def main(args):
    # Setup device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Create DataLoaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        batch_size=args.batch_size
    )

    # Create model
    model = model_builder.create_resnet18k(k=args.k, num_classes=10).to(DEVICE)

    # Set loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Start training with help from engine.py
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        device=DEVICE
    )

    # Create directories if they don't exist
    model_dir = os.path.join(args.run_dir, "models")
    results_dir = os.path.join(args.run_dir, "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save the model
    utils.save_model(
        model=model,
        target_dir=model_dir,
        model_name=f"resnet18k_{args.k}.pth"
    )

    # Save results
    torch.save(results, os.path.join(results_dir, f"resnet18k_{args.k}_results.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18k model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--k", type=int, required=True, help="Width parameter for ResNet18k")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory to save results and models")
    
    args = parser.parse_args()
    main(args)