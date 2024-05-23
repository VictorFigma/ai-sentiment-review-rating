import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import json

def to_lowercase(text):
    return str(text).lower()

# Dataset
class ReviewDataset(Dataset):
    def __init__(self, summaries, texts, labels, tokenizer, max_len):
        self.summaries = summaries
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        summary = self.summaries[idx]
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            summary,
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Check for GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

# Load data
train_data = pd.read_csv('train/train.csv', sep=';')

train_data['Summary'] = train_data['Summary'].apply(to_lowercase)
train_data['Text'] = train_data['Text'].apply(to_lowercase)

# Adjust labels to [0, 4]
train_data['Score'] = train_data['Score'] - 1

# Train
model_name = 'bert-base-uncased'
max_len = 240
batch_size = 8
epochs = 3
tokenizer = BertTokenizer.from_pretrained(model_name)

train_dataset = ReviewDataset(
    summaries=train_data['Summary'].tolist(),
    texts=train_data['Text'].tolist(),
    labels=train_data['Score'].tolist(),
    tokenizer=tokenizer,
    max_len=max_len
)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=250,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=20,
    no_cuda=not torch.cuda.is_available()
)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)
model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Load data
test_data = pd.read_csv('test/test.csv', sep=';')
test_data['Summary'] = test_data['Summary'].apply(to_lowercase)
test_data['Text'] = test_data['Text'].apply(to_lowercase)

test_dataset = ReviewDataset(
    summaries=test_data['Summary'].tolist(),
    texts=test_data['Text'].tolist(),
    labels=[0] * len(test_data),
    tokenizer=tokenizer,
    max_len=max_len
)

test_loader = DataLoader(test_dataset, batch_size=batch_size)

predictions = []
model.eval()

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(preds)

# Adjust predictions back to original range [1, 5]
predictions = [int(p + 1) for p in predictions]

# Save to JSON
output = {'target': dict(zip(test_data['Test_id'], predictions))}
with open('predictions/predictions.json', 'w') as f:
    json.dump(output, f)