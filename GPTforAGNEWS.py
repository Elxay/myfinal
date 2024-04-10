import torch
import csv
from transformers import GPT2Tokenizer
from transformers import GPT2ForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm import tqdm
import os


# 读取AG_NEWS CSV文件
def read_ag_news_csv(file_path):
    labels, texts = [], []
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            labels.append(int(row[0]) - 1)  # 类别是1-4，调整为0-3
            text = row[1] + " " + row[2]  # 标题和描述
            texts.append(text)
    return labels, texts


# 数据预处理
def preprocess_data(tokenizer, texts, labels, max_length=512):
    tokenizer.pad_token = tokenizer.eos_token  # 使用eos_token作为pad_token
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    labels = torch.tensor(labels)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks, labels


# 路径设置
train_file_path = 'train.csv'
test_file_path = 'test.csv'

# 加载tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('pretrainedmodels/myGPT2/')

# 使用eos_token作为pad_token
tokenizer.pad_token = tokenizer.eos_token


# 读取并预处理数据
train_labels, train_texts = read_ag_news_csv(train_file_path)
train_input_ids, train_attention_masks, train_labels = preprocess_data(tokenizer, train_texts, train_labels)

# 创建TensorDataset和DataLoader
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=4)

# 加载模型
model = GPT2ForSequenceClassification.from_pretrained('pretrainedmodels/myGPT2/', num_labels=4)
# 确保模型接受pad_token_id作为有效输入
model.config.pad_token_id = tokenizer.pad_token_id


# 训练设置
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average Training Loss: {avg_train_loss:.2f}")

# 保存模型
model_save_path = 'myGptModel'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
