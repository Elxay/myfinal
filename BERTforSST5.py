import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import json
from tqdm import tqdm

class SentimentDataset(Dataset):
    """自定义的情感分析数据集类"""
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts, self.labels = self.read_data(file_path)
        self.max_length = max_length

    def read_data(self, file_path):
        """读取jsonl格式的文件"""
        texts, labels = [], []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                texts.append(data['text'])
                labels.append(data['label'])
        return texts, labels

    def __len__(self):
        """返回数据集的大小"""
        return len(self.texts)

    def __getitem__(self, idx):
        """获取数据集的一个元素"""
        text = self.texts[idx]
        label = self.labels[idx]
        # 使用tokenizer处理文本
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze(0)  # 移除批次维度
        attention_mask = inputs['attention_mask'].squeeze(0)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(label)}

# tokenizer实例
tokenizer = BertTokenizer.from_pretrained('pretrainedmodels/myBERT/')
model = BertForSequenceClassification.from_pretrained('pretrainedmodels/myBERT/', num_labels=5)  # 假设有5个类别
# 创建数据集实例
train_dataset = SentimentDataset(file_path='train.jsonl', tokenizer=tokenizer)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 训练设置
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        model.zero_grad()
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss:.2f}")


# 保存模型
model_save_path = 'myBertforSSTmodel'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)