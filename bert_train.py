import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from nltk.corpus import wordnet

# Load dataset
df = pd.read_csv('statements_dataset.csv')

# Map the correctness labels to numerical values
df['correctness'] = df['correctness'].map({'correct': 1, 'incorrect': 0})

# Split the dataset into features (X) and target (y)
X = df['statement'].tolist()
y = df['correctness'].tolist()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Tokenize the data
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

class StatementDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the dataset and dataloaders
train_dataset = StatementDataset(train_encodings, y_train)
test_dataset = StatementDataset(test_encodings, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay = 0.01)

# Set up the training loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

best_val_loss = float('inf')

model.train()
for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1} loss: {loss.item()}')

    # Validation
    model.eval()
    val_loss = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            val_loss += outputs.loss.item()
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(test_loader)
    print(f'Validation loss: {val_loss}')
    
    # Save the model if validation loss has decreased
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained('./best_model')
        tokenizer.save_pretrained('./best_model')
        print(f'Saving model with validation loss: {best_val_loss}')
    
    model.train()

# Final evaluation with adjusted threshold
model.eval()
predictions, true_labels = [], []

threshold = 0.9  # Adjust this threshold to increase sensitivity

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions.extend(np.where(probs[:, 1] >= threshold, 1, 0))
        true_labels.extend(labels.cpu().numpy())

# Print evaluation metrics
accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions)


print(f'Accuracy: {accuracy}')
print(report)

# Save the model and tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
