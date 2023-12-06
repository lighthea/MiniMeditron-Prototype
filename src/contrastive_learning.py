from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from torch.utils.data import DataLoader
import json


def parse_guideline(text):
    json_obj = json.load(text)
    
    for symptom in json_obj[] 


model = SentenceTransformer('all-MiniLM-L6-v2')
train_examples = [
    InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
    InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
train_loss = losses.ContrastiveLoss(model=model)

model.fit([(train_dataloader, train_loss)], show_progress_bar=True)
