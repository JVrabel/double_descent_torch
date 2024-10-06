import subprocess
import os
from datetime import datetime
from multiprocessing import Pool

def train_model(args):
    k, epochs, batch_size, learning_rate, run_dir, train_script = args
    command = [
        "python", train_script,
        "--epochs", str(epochs),
        "--k", str(k),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--run_dir", run_dir  # Add this line to pass run_dir to train.py
    ]
    subprocess.run(command, check=True)

def main():
    # Setup hyperparameters
    EPOCHS = 10000
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.0001
    K_VALUES = range(1, 65)  # [1, 2, 3, ..., 10]
    NUM_PROCESSES = 5  # Fixed number of subprocesses

    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create a unique directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, f"runs/run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save run configuration
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"K Values: {list(K_VALUES)}\n")
        f.write(f"Number of Subprocesses: {NUM_PROCESSES}\n")

    # Get the absolute path to train.py
    train_script = os.path.join(project_root, "src", "train.py")

    # Prepare arguments for each training run
    args_list = [(k, EPOCHS, BATCH_SIZE, LEARNING_RATE, run_dir, train_script) for k in K_VALUES]

    # Run training processes in parallel
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(train_model, args_list)

if __name__ == "__main__":
    main()