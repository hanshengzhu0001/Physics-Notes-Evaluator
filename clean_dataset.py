import pandas as pd

# Read the CSV file
df = pd.read_csv('statements_dataset.csv')

# Inspect the DataFrame for any invalid entries
print(df.head())

# Define a function to clean and validate the 'statement' field
def clean_statement(statement):
    # Remove unwanted characters or patterns (e.g., serial numbers, extra quotation marks)
    statement = statement.replace('"', '')  # Remove double quotes
    # Add more cleaning rules as needed
    return statement

# Apply the cleaning function to the 'statement' column
df['statement'] = df['statement'].apply(clean_statement)

# Drop any rows that may have empty or invalid 'statement' fields
df = df.dropna(subset=['statement'])

# Check for duplicates again and remove them
df = df.drop_duplicates()

# Save the cleaned DataFrame to a new CSV file
df.to_csv('statements_dataset.csv', index=False)

# Display the cleaned DataFrame
print(df.head())
