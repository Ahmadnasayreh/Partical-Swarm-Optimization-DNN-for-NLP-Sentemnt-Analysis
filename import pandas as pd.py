import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import numpy as np

# Load the dataset
df = pd.read_csv('C:/Users/USER/Desktop/NewReasearch/sent/pytourch/DataForTraining_data.csv')

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.20, random_state=42)

# Convert text data to numerical features using CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

# Apply TF-IDF transformation to the numerical features
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Apply SMOTE to balance the data
smote = SMOTE(random_state=9)
X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train.ravel())

# After SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_train_res.toarray(), y_train_res, test_size=0.20, random_state=0)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Correctly convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
batch_size = 42
# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

# Neural Network class
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_neurons):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, num_neurons)
        self.fc2 = nn.Linear(num_neurons, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, len(np.unique(y_train)))
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = self.fc6(x)
        return x

# This list will hold the best fitness score at each iteration
convergence_curve = [np.inf]

def de_callback(xk, convergence):
    # Evaluate the current fitness
    f_val = de_fitness_function(xk)
    # Save the best fitness: either the current one or the previous best
    convergence_curve.append(min(f_val, convergence_curve[-1]))
    print(f"Current best fitness: {convergence_curve[-1]}")
    return False  # Return False to continue the optimization

# Fitness function for DE
def de_fitness_function(hyperparameters):
    learning_rate, num_neurons = hyperparameters
    num_neurons = int(num_neurons)  # Ensure the number of neurons is an integer
    
    model = NeuralNet(X_train.shape[1], num_neurons).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(2):  # Limited epochs for faster evaluation
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    return val_loss / len(val_loader)

# Define bounds for DE (learning rate and number of neurons)
de_bounds = [(0.0001, 0.01), (20, 1000)]

# Run DE optimization without multiprocessing
result = differential_evolution(
    func=de_fitness_function,
    bounds=de_bounds,
    strategy='best1bin',
    maxiter=20,
    popsize=10,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    disp=True,
    callback=de_callback,
    seed=42,
    workers=1  # Use 1 to avoid using multiprocessing
)


# After optimization is complete, plot the convergence curve
plt.figure(figsize=(10, 6))
plt.plot(convergence_curve[1:], marker='o')  # Skip the first np.inf value
plt.title('DE Convergence Curve')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness Score (Validation Loss)')
plt.grid(True)
plt.show()

# Print the best fitness score
best_fitness_score = convergence_curve[-1]
print(f"Best Fitness Score (Validation Loss) from DE: {best_fitness_score}")
# Best hyperparameters from DE optimization
best_learning_rate, best_num_neurons = result.x
print(f"Best hyperparameters from DE optimization: Learning Rate = {best_learning_rate}, Number of Neurons = {int(best_num_neurons)}")
print(f"Best fitness (validation loss) from DE optimization: {result.fun}")

# Create the final model instance with the best hyperparameters
final_model = NeuralNet(X_train.shape[1], int(best_num_neurons)).to(device)

# Define the loss function and optimizer with the best learning rate
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_learning_rate)

# Train the deep neural network
num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    epoch_train_correct = 0
    epoch_val_correct = 0

    final_model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() * inputs.size(0)
        epoch_train_correct += torch.sum(torch.argmax(outputs, dim=1) == targets).item()

    epoch_train_loss /= len(X_train_tensor)
    epoch_train_accuracy = epoch_train_correct / len(X_train_tensor)

    final_model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = final_model(inputs)
            val_loss = criterion(outputs, targets)
            epoch_val_loss += val_loss.item() * inputs.size(0)
            epoch_val_correct += torch.sum(torch.argmax(outputs, dim=1) == targets).item()

    epoch_val_loss /= len(X_test_tensor)
    val_accuracy = epoch_val_correct / len(X_test_tensor)
    
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accuracies.append(epoch_train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
          f"Train Accuracy: {epoch_train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

# After training, evaluate the model's performance using a confusion matrix and classification report


# Generate predictions using the final model for confusion matrix and classification report
final_model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for inputs, targets in val_loader:
        outputs = final_model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Confusion Matrix
conf_mat = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for DE-DNN')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(all_targets, all_preds))

# Plotting training and validation losses
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
