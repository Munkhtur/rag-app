import pandas as pd
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses



df = pd.read_csv('./history-train-data.csv', on_bad_lines='error')

print("---------------df done -----------------------")
train_examples = [
    InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label'])
    for index, row in df.iterrows()
]

print(train_examples[0])

print("--------------- train example -----------------------")

# Load the model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

print("--------------- model -----------------------")
# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)  # For binary classification

# Fine-tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

print("--------------- fit  -----------------------")
# Save the fine-tuned model
model.save('history-fine-tuned')