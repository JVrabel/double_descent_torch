import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import random
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src import data_setup, engine, model_builder

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size
    )

    model = model_builder.create_resnet18k(k=args.k, num_classes=10).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    results_dir = os.path.join(args.run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"resnet18k_{args.k}_seed_{args.seed}_results.json")

    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        device=DEVICE,
        results_file=results_file,
        k=args.k,
        print_every=args.print_every,
        seed=args.seed
    )

    print(f"Training completed. Results saved in {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18k model")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--k", type=int, default=1, help="Width factor for ResNet18k")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=1000, help="Batch size for testing")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--run_dir", type=str, default="./runs", help="Directory to save results")
    parser.add_argument("--print_every", type=int, default=10, help="Print frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)