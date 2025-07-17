import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from training.dataset.dataloader import SimpleHDF5Dataset
from training.models.gcn_bert.util import collate_fn_gcnbert
from training.models.gcn_bert.gcn_bert import GCN_BERT


# === Paths
train_path     = "/data/cristian/paper_2025/Testing/ISLR/WLASL/WLASL100/WLASL100_135-Train.hdf5"
val_path       = "/data/cristian/paper_2025/Testing/ISLR/WLASL/WLASL100/WLASL100_135-Val.hdf5"
test_path      = "/data/cristian/paper_2025/Testing/ISLR/WLASL/WLASL100/WLASL100_135-Test.hdf5"

map_label_path = "/data/cristian/paper_2025/Testing/ISLR/WLASL/WLASL100/wlasl_100_maplabels.json"


# === Loaders
train_loader = DataLoader(SimpleHDF5Dataset(train_path,map_label_path,augmentation=True,noise_std=0.01), batch_size=8, shuffle=True, collate_fn=collate_fn_gcnbert)
val_loader   = DataLoader(SimpleHDF5Dataset(val_path,map_label_path),   batch_size=8, collate_fn=collate_fn_gcnbert)
test_loader  = DataLoader(SimpleHDF5Dataset(test_path,map_label_path),   batch_size=8, collate_fn=collate_fn_gcnbert)

# === Model
model = GCN_BERT(num_classes=135, hidden_features=2, seq_len=50, num_joints=135,nhead=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
import os

# === Configuración
EPOCHS = 2000
PATIENCE = 100

BEST_MODEL_PATH = "../../results/models/gcn_bert/wlasl_best.pth"
os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_preds, train_targets = 0.0, [], []

    loop = tqdm(train_loader, total=len(train_loader), ncols=100, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for x, y, mask, _ in loop:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        out = model(x, mask)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_preds.extend(out.argmax(dim=1).cpu().numpy())
        train_targets.extend(y.cpu().numpy())

        acc = accuracy_score(train_targets, train_preds)
        loop.set_postfix(loss=loss.item(), acc=acc)

    epoch_train_acc = accuracy_score(train_targets, train_preds)

    # === Validation
    model.eval()
    val_loss, val_preds, val_targets = 0.0, [], []
    with torch.no_grad():
        val_loop = tqdm(val_loader, total=len(val_loader), ncols=100, desc=f"Epoch {epoch+1}/{EPOCHS} [Val  ]")
        for x, y, mask, _ in val_loop:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            out = model(x, mask)
            loss = criterion(out, y)

            val_loss += loss.item()
            val_preds.extend(out.argmax(dim=1).cpu().numpy())
            val_targets.extend(y.cpu().numpy())

            acc_val = accuracy_score(val_targets, val_preds)
            val_loop.set_postfix(loss=loss.item(), acc=acc_val)

    epoch_val_acc = accuracy_score(val_targets, val_preds)

    # === Early stopping check
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✅ New best model saved! Val Acc: {best_val_acc:.4f}")
    else:
        epochs_no_improve += 1
        print(f"⏳ No improvement for {epochs_no_improve} epochs")

    if epochs_no_improve >= PATIENCE:
        print(f"🛑 Early stopping at epoch {epoch+1}")
        break

# === Final Test
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()
test_loss, test_preds, test_targets = 0.0, [], []
with torch.no_grad():
    test_loop = tqdm(test_loader, total=len(test_loader), ncols=100, desc="[TEST]")
    for x, y, mask, _ in test_loop:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        out = model(x, mask)
        loss = criterion(out, y)

        test_loss += loss.item()
        test_preds.extend(out.argmax(dim=1).cpu().numpy())
        test_targets.extend(y.cpu().numpy())

        acc_test = accuracy_score(test_targets, test_preds)
        test_loop.set_postfix(loss=loss.item(), acc=acc_test)

final_test_acc = accuracy_score(test_targets, test_preds)
print(f"\n✅ [TEST FINAL] Loss: {test_loss:.4f} | Accuracy: {final_test_acc:.4f}")
