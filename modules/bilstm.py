import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=3, dropout_rate=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hn, cn) = self.lstm(packed)
        else:
            _, (hn, cn) = self.lstm(x)
        hidden_concat = torch.cat((hn[0], hn[1]), dim=1)
        out = self.fc(self.dropout(hidden_concat))
        return out

def run_training(
    X_features: np.ndarray,
    y_labels: np.ndarray,
    lengths: np.ndarray = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    patience: int = 3,
    min_delta: float = 0.0,
    epoch_callback=None
) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    num_classes = len(le.classes_)

    idx = np.arange(len(X_features))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y_encoded
    )

    X_train, X_test = X_features[idx_train], X_features[idx_test]
    y_train, y_test = y_encoded[idx_train], y_encoded[idx_test]
    len_train = lengths[idx_train] if lengths is not None else None
    len_test  = lengths[idx_test]  if lengths is not None else None

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    if lengths is not None:
        len_train_t = torch.tensor(len_train, dtype=torch.long)
        len_test_t  = torch.tensor(len_test,  dtype=torch.long)
        train_ds = TensorDataset(X_train_t, y_train_t, len_train_t)
        test_ds  = TensorDataset(X_test_t,  y_test_t,  len_test_t)
    else:
        train_ds = TensorDataset(X_train_t, y_train_t)
        test_ds  = TensorDataset(X_test_t,  y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = BiLSTMClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc':  [], 'test_acc':  []
    }

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0
    stopped_early = False

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            if lengths is not None:
                inputs, labels, batch_lens = batch
                batch_lens = batch_lens.to(device)
            else:
                inputs, labels = batch
                batch_lens = None
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, batch_lens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc  = correct / total

        model.eval()
        test_loss, correct_test, total_test = 0.0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                if lengths is not None:
                    inputs, labels, batch_lens = batch
                    batch_lens = batch_lens.to(device)
                else:
                    inputs, labels = batch
                    batch_lens = None
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, batch_lens)
                loss = criterion(outputs, labels)
                test_loss    += loss.item()
                _, predicted  = torch.max(outputs.data, 1)
                total_test   += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_acc  = correct_test / total_test

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        # --- Early stopping check (based on validation loss) ---
        if test_loss < best_val_loss - min_delta:
            best_val_loss = test_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_no_improve = 0
            improved_marker = " *"
        else:
            epochs_no_improve += 1
            improved_marker = ""

        log = (
            f"Epoch [{epoch+1:>2}/{epochs}] "
            f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
            f"| Val Loss: {test_loss:.4f}, Val Acc: {test_acc:.4f}{improved_marker}"
        )
        if epoch_callback:
            epoch_callback(epoch + 1, epochs, log)

        if epochs_no_improve >= patience:
            stopped_early = True
            stop_log = (
                f"Early stopping di epoch {epoch+1} "
                f"(tidak ada peningkatan Val Loss selama {patience} epoch berturut-turut, "
                f"terbaik: epoch {best_epoch} dengan Val Loss {best_val_loss:.4f})"
            )
            if epoch_callback:
                epoch_callback(epoch + 1, epochs, stop_log)
            break

    # Muat kembali bobot terbaik (bukan bobot epoch terakhir) sebelum evaluasi akhir
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            if lengths is not None:
                inputs, labels, batch_lens = batch
                batch_lens = batch_lens.to(device)
            else:
                inputs, labels = batch
                batch_lens = None
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, batch_lens)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    report = classification_report(
        all_targets, all_preds,
        target_names=le.classes_,
        output_dict=True
    )
    cm = confusion_matrix(all_targets, all_preds)

    return {
        'model':        model,
        'label_encoder': le,
        'history':      history,
        'report':       report,
        'confusion_matrix': cm,
        'class_names':  list(le.classes_),
        'n_train':      len(X_train),
        'n_test':       len(X_test),
        'y_test': all_targets,
        'y_pred': all_preds,
        'best_epoch':   best_epoch,
        'stopped_early': stopped_early
    }