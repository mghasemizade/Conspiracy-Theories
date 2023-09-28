from imports import *

model_directory = 'classifier' #path to our best classifier model
# Load the tokenizer and model from the directory
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForSequenceClassification.from_pretrained(model_directory)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    do_predict=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
)


def predict(text, threshold): #the function takes a value for threshold for the probability of the predicted labels
    # Preprocess the text
    inputs = tokenizer([text], truncation=True, padding=True, return_tensors='pt')
    inputs = Dataset.from_dict({k: v.tolist() for k, v in inputs.items()})  # Convert to Dataset object

    # Get the predictions
    predictions, _, _ = trainer.predict(inputs)

    # Apply the softmax function to get probabilities
    probabilities = torch.sigmoid(torch.from_numpy(predictions)).numpy()
    #print(probabilities)


    # Initialize a DataFrame to hold the labels
    label = (probabilities[:, 1] > threshold).astype(int)
    #print(label)

    return label
