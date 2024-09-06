import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# Set the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the enhanced classifier model
class EnhancedClassifier(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)
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

# Initialize the classifier model
input_dim = sbert_model.get_sentence_embedding_dimension()
classifier_model = EnhancedClassifier(input_dim).to(device)

# Load the trained model weights
classifier_model.load_state_dict(torch.load('./best_model.pt'))
classifier_model.eval()

# Sample new statements for prediction
new_statements = [
    "The shortest distance from the initial to the final position of an object in a specific direction is called displacement.",
    "The rate of change of velocity of an object is known as acceleration.",
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
    "The safety net increases the force by increasing the time interval, as per the equation F = Δp/Δt."
]

# Encode the new statements
new_embeddings = sbert_model.encode(new_statements, convert_to_tensor=True).to(device)

# Make predictions
with torch.no_grad():
    outputs = classifier_model(new_embeddings)
    predictions = torch.argmax(outputs, dim=-1)

# Convert predictions to labels
predicted_labels = predictions.cpu().numpy()

# Print the results
for statement, label in zip(new_statements, predicted_labels):
    print(f"Statement: {statement}\nPredicted Label: {'Correct' if label == 1 else 'Incorrect'}\n")
