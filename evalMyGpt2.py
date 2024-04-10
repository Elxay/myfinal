from transformers import GPT2Tokenizer
from transformers import GPT2ForSequenceClassification
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np



model = GPT2ForSequenceClassification.from_pretrained('myGptModel')
tokenizer = GPT2Tokenizer.from_pretrained('myGptModel')

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


test_file_path = 'test.csv'

# 使用eos_token作为pad_token
tokenizer.pad_token = tokenizer.eos_token


# 读取并预处理数据
test_labels, test_texts = read_ag_news_csv(test_file_path)
test_input_ids, test_attention_masks, test_labels_tensor = preprocess_data(tokenizer, test_texts, test_labels)

# 使用DataLoader
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=4)  # 可以调整batch_size

# 转换为评估模式
model.eval()

# 初始化用于保存预测和真实标签的列表
true_labels = []
pred_labels = []

# 检查CUDA是否可用，并据此将模型移动到GPU或保持在CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 禁用梯度计算
with torch.no_grad():
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # 进行预测
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        # 获取最大概率的索引作为预测标签
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # 更新列表
        true_labels.extend(b_labels.cpu().numpy())
        pred_labels.extend(predictions.cpu().numpy())

# 计算评价标准
accuracy = accuracy_score(true_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')