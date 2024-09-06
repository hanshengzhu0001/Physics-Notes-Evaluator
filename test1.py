import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Load the trained model
model = BertForSequenceClassification.from_pretrained('./best_model')
model.to(device)
model.eval()

# Sample new statements for prediction
new_statements = [
    "The shortest distance from the initial to the final position of an object in a specific direction is called displacement.",
    "The rate of change of velocity of an object acceleration.",
    "An object in motion will remain in motion, and an object at rest will remain at rest unless acted upon by a net external force.",
    "Force is equal to the mass of an object multiplied by its acceleration (F = ma).",
    "For every action, there is an equal and opposite reaction.",
    "The energy an object possesses due to its motion is called kinetic energy, calculated as KE = 0.5 * m * v^2.",
    "The energy stored in an object due to its position or state is known as potential energy, such as gravitational potential energy (PE = mgh).",
    "The work done on an object is equal to the change in its kinetic energy.",
    "The change in momentum of an object when a force is applied over a time interval is calculated as Impulse = F * Δt.",
    "The total energy of an isolated system remains constant; energy can neither be created nor destroyed, only transformed from one form to another.",
    "The shortest distance from the initial to the final position of an object is known as displacement, regardless of direction.",
    "The rate of change of speed of an object is called deceleration.",
    "An object in motion will stop immediately unless acted upon by a net external force.",
    "Force is equal to the mass of an object divided by its acceleration (F = m/a).",
    "For every action, there is an unequal and opposite reaction.",
    "The energy an object possesses due to its motion is called kinetic energy, calculated as KE = m * v^2.",
    "The energy stored in an object due to its position or state is known as kinetic energy (KE = mgh).",
    "The work done on an object is equal to the change in its temperature.",
    "The change in velocity of an object when a force is applied over a time interval is calculated as Impulse = F + t.",
    "The total energy of an isolated system can be created and destroyed.",
    "The safety net increases the force by decreasing the time interval, as per the equation F = Δp/Δt."
]

# Tokenize the new statements
encodings = tokenizer(new_statements, truncation=True, padding=True, return_tensors='pt').to(device)

# Make predictions
with torch.no_grad():
    outputs = model(**encodings)
    probs = torch.softmax(outputs.logits, dim=-1)
    predictions = torch.argmax(probs, dim=-1)
    confidence_scores = probs.max(dim=-1).values

# Convert predictions to labels
predicted_labels = predictions.cpu().numpy()
confidence_scores = confidence_scores.cpu().numpy()

# Print the results with confidence levels
for statement, label, confidence in zip(new_statements, predicted_labels, confidence_scores):
    print(f"Statement: {statement}\nPredicted Label: {'Correct' if label == 1 else 'Incorrect'}\nConfidence Level: {confidence:.4f}\n")
