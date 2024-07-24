# BERT-based Text Classification

This project demonstrates how to use a pre-trained BERT model for text classification tasks using PyTorch and Hugging Face's Transformers library.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Installation

To get started, you need to install the required libraries. You can do this using `pip`:

```sh
pip install torch transformers scikit-learn
```

Ensure you have a GPU available and configured if you plan to train the model on a GPU for faster processing.

## Dataset Preparation

Replace the placeholder data with your actual dataset. The dataset should consist of a list of texts and corresponding labels.

```python
train_texts = ["example text 1", "example text 2"]
train_labels = [0, 1]
test_texts = ["example text 3", "example text 4"]
test_labels = [0, 1]
```

Split your dataset into training and validation sets using `train_test_split`.

## Training the Model

The model is defined using the BERT architecture from the Hugging Face Transformers library. The script includes tokenization, model definition, training, and validation processes.

Run the script `train_model.py` to start the training process:

```python
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the text classification dataset
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Dummy data for illustration; replace with actual data
train_texts = ["example text 1", "example text 2"]
train_labels = [0, 1]
test_texts = ["example text 3", "example text 4"]
test_labels = [0, 1]

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

# Create data loaders
batch_size = 32
train_dataset = TextClassificationDataset(train_texts, train_labels)
val_dataset = TextClassificationDataset(val_texts, val_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the text classification model
class TextClassificationModel(torch.nn.Module):
    def __init__(self):
        super(TextClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, len(set(train_labels)))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return outputs

# Initialize the model, optimizer, and loss function
model = TextClassificationModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / len(val_labels)
    print(f'Validation Accuracy: {accuracy:.4f}')
```

## Evaluation

After training, evaluate the model on the test set to measure its performance. The evaluation script calculates the accuracy of the model.

```python
# Evaluate the model on the test set
test_dataset = TextClassificationDataset(test_texts, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_correct = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, dim=1)
        test_correct += (predicted == labels).sum().item()

test_accuracy = test_correct / len(test_labels)
print(f'Test Accuracy: {test_accuracy:.4f}')
```

## Results

After running the training and evaluation scripts, you should see the training loss and validation accuracy for each epoch, followed by the test accuracy. Example output:

```
Epoch 1, Loss: 0.5678
Validation Accuracy: 0.7500
...
Test Accuracy: 0.8000
```

## Acknowledgements

This project uses the [Transformers library by Hugging Face](https://huggingface.co/transformers/) and [PyTorch](https://pytorch.org/).

## License

This project is licensed under the MIT License. See the LICENSE file for details.
