from transformers import BertTokenizer, BertForSequenceClassification
import torch

def main():
    # Load the pre-trained BERT tokenizer and model for sequence classification (sentiment analysis)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Input text for sentiment analysis
    text = "I love using transformers for NLP tasks!"

    # Tokenize the input text and convert it to a format BERT can understand
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Run the text through the BERT model
    with torch.no_grad():
        outputs = model(**inputs)

    # The model's output logits (unnormalized predictions)
    logits = outputs.logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Get the predicted label (0 = negative, 1 = positive)
    predicted_label = torch.argmax(probabilities).item()

    # Map predicted label to sentiment
    sentiment = "positive" if predicted_label == 1 else "negative"

    print(f"Text: {text}")
    print(f"Predicted sentiment: {sentiment}")


if __name__ == '__main__':
    main()

