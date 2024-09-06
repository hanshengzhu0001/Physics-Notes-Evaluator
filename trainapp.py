import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification, pipeline, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import logging
from flask import Flask

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "Machine learning code is running"

def run_ml_code():
    try:
        logger.info("Loading dataset...")
        # Load dataset
        df = pd.read_csv('modified_physics_dataset.csv')

        # Map the correctness labels to numerical values
        df['correctness'] = df['correctness'].map({'correct': 1, 'incorrect': 0})

        logger.info("Initializing NER model and tokenizer...")
        # Initialize the NER model and tokenizer
        ner_model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
        ner_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        ner_pipeline = pipeline('ner', model=ner_model, tokenizer=ner_tokenizer)

        # Extract entities using NER
        def extract_entities(text):
            ner_results = ner_pipeline(text)
            entities = " ".join([f"{ent['word']}({ent['entity']})" for ent in ner_results])
            return entities

        df['entities_str'] = df['statement'].apply(extract_entities)

        # Combine statements with their extracted entities
        df['input_text'] = df['statement'] + " " + df['entities_str']

        # Split the dataset into features (X) and target (y)
        X = df['input_text'].tolist()
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
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

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
            logger.info(f'Epoch {epoch + 1} loss: {loss.item()}')

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
            logger.info(f'Validation loss: {val_loss}')

            # Save the model if validation loss has decreased
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained('./best_model')
                tokenizer.save_pretrained('./best_model')
                logger.info(f'Saving model with validation loss: {best_val_loss}')

            model.train()

        # Final evaluation with adjusted threshold
        model.eval()
        predictions, true_labels = [], []

        threshold = 0.76  # Adjust this threshold to increase sensitivity

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

        logger.info(f'Accuracy: {accuracy}')
        logger.info(report)

        # Save the model and tokenizer
        model.save_pretrained('./saved_model')
        tokenizer.save_pretrained('./saved_model')

    except Exception as e:
        logger.error(f"Error during processing: {e}")

if __name__ == '__main__':
    run_ml_code()
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
