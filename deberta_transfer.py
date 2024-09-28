import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import torch
import pandas as pd
import random
import numpy as np
from transformers import DebertaTokenizer, DebertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import csv
import datetime
import wandb


wandb.login()
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_file_path = 'thesis-codebase/data/env_train.tsv'
val_file_path = 'thesis-codebase/data/env_dev.tsv'
test_file_path = 'thesis-codebase/data/cw_diabetes_test.tsv'

train_filename = os.path.basename(train_file_path)
train_prefix = train_filename.split('_')[0]

# Load data
train_data = pd.read_csv(train_file_path, sep='\t', header=0)
val_data = pd.read_csv(val_file_path, sep='\t', header=0, quoting=csv.QUOTE_NONE)  
test_data = pd.read_csv(test_file_path, sep='\t', header=0, quoting=csv.QUOTE_NONE)

# Preprocessing: Drop NaN and encode labels
train_data = train_data[['text', 'label']].dropna()
val_data = val_data[['text', 'label']].dropna() 
test_data = test_data[['text', 'label']].dropna()

label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])
val_data['label'] = label_encoder.fit_transform(val_data['label'])
test_data['label'] = label_encoder.fit_transform(test_data['label'])

# Tokenizer initialization
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Dataset creation
train_dataset = Dataset.from_dict({'text': train_data['text'], 'label': train_data['label']})
val_dataset = Dataset.from_dict({'text': val_data['text'], 'label': val_data['label']})  
test_dataset = Dataset.from_dict({'text': test_data['text'], 'label': test_data['label']})

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)  

# Training arguments
training_args = TrainingArguments(
    output_dir='./thesis-codebase/results',
    num_train_epochs=100,
    evaluation_strategy="epoch",  
    save_strategy="epoch",  
    logging_strategy="epoch", 
    save_total_limit=1,  
    metric_for_best_model="f1", 
    greater_is_better=True,  
    do_eval=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    report_to="wandb",
    logging_dir='./thesis-codebase/logs', 
)

# Model initialization
model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=2)

# Define metrics
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = classification_report(labels, preds, output_dict=True, zero_division=0)['weighted avg'].values()
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset, 
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]  # Early stopping with patience of 10 epochs
)

# Train the model
trainer.train()

# Evaluate on the test set
results = trainer.evaluate(test_dataset)
print(results)

# Save evaluation results
with open(f'thesis-codebase/metrics/{train_prefix}_deberta_evaluation_results_{current_time}.txt', 'w') as f:
  f.write(str(results))  

# Predictions and classification report
preds_output = trainer.predict(test_dataset)
preds = preds_output.predictions.argmax(-1)
print(classification_report(test_data['label'], preds))
with open(f'thesis-codebase/metrics/{train_prefix}_deberta_evaluation_results_{current_time}.txt', 'w') as f:
  f.write(classification_report(test_data['label'], preds))  

# Compute and print confusion matrix
conf_matrix = confusion_matrix(test_data['label'], preds)
print("Confusion Matrix:")
print(conf_matrix)

with open(f'thesis-codebase/metrics/{train_prefix}_deberta_evaluation_results_{current_time}_cm.txt', 'w') as f:
  f.write(str(conf_matrix))  

wandb.finish()
