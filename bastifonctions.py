# from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import torch
import os
import json

class Text2TextDataset(Dataset):
    def __init__(self, input_encodings, label_encodings):
        self.input_encodings = input_encodings
        self.label_encodings = label_encodings

    def __len__(self):
        return len(self.input_encodings['input_ids'])

    def __getitem__(self, idx):
        input_item = {key: val[idx] for key, val in self.input_encodings.items()}
        label_item = {key: val[idx] for key, val in self.label_encodings.items()}
        input_item['labels'] = label_item['input_ids']
        return input_item
    
def fine_tune_model(model, train_data, labels, tokenizer, retriever, epochs =5, batch_size=2, padding=4096):
    tokenizer.pad_token = tokenizer.eos_token
    train = tokenizer(train_data, truncation=True, padding='max_length', max_length=padding, return_tensors='pt')
    #train = chunk_and_encode(train_data, tokenizer, 1024, 200)
    labels_encodings = tokenizer(labels, truncation=False, padding='max_length', max_length=50, return_tensors='pt')  # Assuming a max_length of 50 for labels, adjust if needed
    train_dataset = Text2TextDataset(train, labels_encodings)
    print([len(t) for t in train['input_ids']])
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_dir='./logs',
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator= data_collator
    )

    trainer.train()    
    
def download_data(data_path):
    text_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    texts = []
    labels = []

    for text_file in text_files:
        with open(os.path.join(data_path, text_file), 'r') as f:
            text = f.read()
            texts.append(text)
            labels.append(os.path.splitext(text_file)[0])  # Remove file extension to get label
    return texts, labels