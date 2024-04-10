import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import json

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
tokenizer = BertTokenizer.from_pretrained('myBertforSSTmodel')
model = BertForSequenceClassification.from_pretrained('myBertforSSTmodel', num_labels=5)  # 假设有5个类别

# 创建数据集实例
test_dataset = SentimentDataset(file_path='test.jsonl', tokenizer=tokenizer)

# 创建DataLoader
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# 检查CUDA是否可用，并据此将模型移动到GPU或保持在CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def evaluate(model, dataloader):
    model.eval()  # 设置为评估模式

    predictions, true_labels = [], []

    for batch in dataloader:
        # 正确地从批次字典中提取数据并发送到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        predictions.extend(np.argmax(logits, axis=1))
        true_labels.extend(label_ids)

    return true_labels, predictions

true_labels, predictions = evaluate(model, test_loader)

# 计算性能指标
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")