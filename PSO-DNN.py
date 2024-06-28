import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# Load the dataset
df = pd.read_csv('C:/Users/USER/Desktop/pytourch/monk_sent_Training_data.csv')

# Preprocess the text data
tokenizer = lambda x: x.split()  # Simple whitespace tokenizer
from collections import Counter
from torchtext.vocab import vocab  # Note the change here
from torchtext._torchtext import Vocab  # This is for the Vocab class

# Existing code to tokenize and count your text
counter = Counter()
for line in df['text']:
    counter.update(tokenizer(line))

# Filter tokens below a certain frequency threshold
min_freq = 1
filtered_counter = Counter({token: count for token, count in counter.items() if count >= min_freq})

# Create a Vocab instance
vocabulary = Vocab(filtered_counter)

def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

# Dataset class
class TextDataset(Dataset):
    def __init__(self, df):
        self.texts = [torch.tensor(text_pipeline(text), dtype=torch.int64) for text in df['text']]
        self.labels = [label for label in df['label']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Define the model
class LSTMGRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim_lstm, hidden_dim_gru, num_classes):
        super(LSTMGRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim_lstm, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.gru = nn.GRU(hidden_dim_lstm, hidden_dim_gru, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim_gru, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout1(lstm_out)
        gru_out, _ = self.gru(lstm_out)
        gru_out = self.dropout2(gru_out)
        out = self.fc(gru_out[:, -1, :])
        return out

# Collate function to pad text sequences
def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
    return pad_sequence(text_list, padding_value=3.0), torch.tensor(label_list, dtype=torch.int64)

# Splitting and creating DataLoaders
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = TextDataset(train_df)
test_dataset = TextDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

# ... Rest of your training code here, including model initialization, loss function, optimizer, training loop, etc.
# ... Continuing from the model definition and DataLoader setup

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Initialize the model
vocab_size = len(vocab)
embed_dim = 100  # You can adjust this
hidden_dim_lstm = 64
hidden_dim_gru = 32
num_classes = df['label'].nunique()
model = LSTMGRUModel(vocab_size, embed_dim, hidden_dim_lstm, hidden_dim_gru, num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=0.00001)

# Training loop
num_epochs = 10
history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct / total
    history['loss'].append(train_loss / len(train_loader))
    history['accuracy'].append(train_accuracy)

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct / total
    history['val_loss'].append(val_loss / len(test_loader))
    history['val_accuracy'].append(val_accuracy)
    scheduler.step(val_loss)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss / len(test_loader):.4f}, Val Acc: {val_accuracy:.2f}%')

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# Generate predictions for confusion matrix
model.eval()
y_pred = []
y_true
