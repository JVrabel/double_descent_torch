import os
import subprocess
import datetime
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product

def train_model(args):
    k, epochs, batch_size, test_batch_size, learning_rate, run_dir, train_script, master_seed = args
    model_seed = master_seed + k  # Calculate model-specific seed
    command = [
        "python", train_script,
        "--epochs", str(epochs),
        "--k", str(k),
        "--batch_size", str(batch_size),
        "--test_batch_size", str(test_batch_size),
        "--learning_rate", str(learning_rate),
        "--run_dir", run_dir,
        "--seed", str(model_seed)
    ]
    subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(description="Control script for running multiple ResNet18k trainings")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel processes to run")
    args = parser.parse_args()

    # Parameters
    K_VALUES = range(1, 65)  # k values from 1 to 10
    EPOCHS = 1000
    BATCH_SIZE = 128
    TEST_BATCH_SIZE = 10000
    LEARNING_RATE = 0.0001
    MASTER_SEEDS = 42 # [42, 43, 44]  # Multiple master seeds

    # Create a unique run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(f"runs/run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Path to the train.py script
    train_script = os.path.abspath("src/train.py")

    # Prepare arguments for each run
    args_list = [
        (k, EPOCHS, BATCH_SIZE, TEST_BATCH_SIZE, LEARNING_RATE, run_dir, train_script, MASTER_SEEDS)
        for k in K_VALUES # product(K_VALUES, MASTER_SEEDS)
    ]

    # Run the training processes in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        executor.map(train_model, args_list)

    print(f"All training runs completed. Results saved in {run_dir}")

if __name__ == "__main__":
    main()