import torch
from tqdm import tqdm
import json

def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, device, results_file, k, print_every, seed):
    results = []
    
    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        
        epoch_results = {
            "epoch": epoch + 1,
            "k": k,
            "seed": seed,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc)
        }
        
        results.append(epoch_results)
        
        save_results(results, results_file)
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch: {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.4f}")

    return results

def save_results(results, results_file):
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += (y_pred.argmax(1) == y).sum().item() / len(y)
    
    return train_loss / len(dataloader), train_acc / len(dataloader)

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            
            test_loss += loss.item()
            test_acc += (y_pred.argmax(1) == y).sum().item() / len(y)
    
    return test_loss / len(dataloader), test_acc / len(dataloader)