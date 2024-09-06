import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('statements_dataset.csv')

# Map the correctness labels to numerical values
df['correctness'] = df['correctness'].map({'correct': 1, 'incorrect': 0})

# Split the dataset into features (X) and target (y)
X = df['statement'].tolist()
y = df['correctness'].tolist()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the data
train_embeddings = sbert_model.encode(X_train, convert_to_tensor=True)
test_embeddings = sbert_model.encode(X_test, convert_to_tensor=True)

class StatementDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __getitem__(self, idx):
        return {'embedding': self.embeddings[idx], 'label': torch.tensor(self.labels[idx], dtype=torch.long)}

    def __len__(self):
        return len(self.labels)

# Create the dataset and dataloaders
train_dataset = StatementDataset(train_embeddings, y_train)
test_dataset = StatementDataset(test_embeddings, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the enhanced classifier model
class EnhancedClassifier(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Set the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initialize the classifier model
input_dim = train_embeddings.shape[1]
classifier_model = EnhancedClassifier(input_dim).to(device)

# Set up the optimizer and loss function
optimizer = optim.Adam(classifier_model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Set up the training loop
best_val_loss = float('inf')

for epoch in range(5):
    classifier_model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        embeddings = batch['embedding'].to(device)
        labels = batch['label'].to(device)
        outputs = classifier_model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1} loss: {loss.item()}')

    # Validation
    classifier_model.eval()
    val_loss = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            outputs = classifier_model(embeddings)
            val_loss += criterion(outputs, labels).item()
            preds = torch.argmax(outputs, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(test_loader)
    print(f'Validation loss: {val_loss}')
    
    # Save the model if validation loss has decreased
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(classifier_model.state_dict(), './best_model.pt')
        print(f'Saving model with validation loss: {best_val_loss}')

# Final evaluation
classifier_model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        embeddings = batch['embedding'].to(device)
        labels = batch['label'].to(device)
        outputs = classifier_model(embeddings)
        preds = torch.argmax(outputs, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Print evaluation metrics
accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions)

print(f'Accuracy: {accuracy}')
print(report)

# Save the model
torch.save(classifier_model.state_dict(), './saved_model.pt')

