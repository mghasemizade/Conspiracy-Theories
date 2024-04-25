import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch


# Specify the path to your model
model_directory = 'classifier'

# Load the tokenizer and model from the directory
model = AutoModel.from_pretrained('classifier')
tokenizer = AutoTokenizer.from_pretrained(model_directory)

def predict(text, threshold=0.5):
    # Tokenize the text with truncation and padding
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
    # Feed the tokenized text to the model
    outputs = model(**inputs)
    # Get the logits from the model output
    logits = outputs.logits
    # Use sigmoid function to get probabilities
    probabilities = torch.sigmoid(logits)
    # Convert tensor to numpy array
    probabilities = probabilities.detach().numpy().tolist()[0]
    # Get the index of the label with the highest score
    max_index = probabilities.index(max(probabilities))
    # Prepare result
    score = round(probabilities[max_index], 2)
    if score >= threshold:
        label = 1
    else:
        label = 0
    return label

# Load your test data
df = pd.read_csv('test_data.csv')

# Use the model to make predictions for all texts in the test data
df['predicted_label'] = df['text'].apply(lambda x: predict(x, threshold=0.5))

# Calculate accuracy and F1 score
accuracy = accuracy_score(df['label'], df['predicted_label'])
f1 = f1_score(df['label'], df['predicted_label'])

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')