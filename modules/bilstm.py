import torch
import torch.nn as nn
import torch.optim as optim
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
                nn.init.kaiming_uniform_(param, nonlinearity='tanh')
            elif 'weight_hh' in name:
                nn.init.kaiming_uniform_(param, nonlinearity='tanh')
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        hidden_concat = torch.cat((hn[0], hn[1]), dim=1)
        out = self.fc(self.dropout(hidden_concat))
        return out

def run_training(
    X_features: np.ndarray,
    y_labels: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    epoch_callback=None
) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=batch_size, shuffle=False)

    model = BiLSTMClassifier(input_dim=768, hidden_dim=128, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc':  [], 'test_acc':  []
    }

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
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

        log = (
            f"Epoch [{epoch+1:>2}/{epochs}] "
            f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
            f"| Val Loss: {test_loss:.4f}, Val Acc: {test_acc:.4f}"
        )
        if epoch_callback:
            epoch_callback(epoch + 1, epochs, log)

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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
    }