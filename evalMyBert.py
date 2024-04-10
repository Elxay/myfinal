import torch
import csv

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import numpy as np

# 加载BERT的分词器和模型
tokenizer = BertTokenizer.from_pretrained('myBertModel')
model = BertForSequenceClassification.from_pretrained('myBertModel')

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
    input_ids = []
    attention_masks = []

    # 对每个文本进行处理
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,  # 要编码的文本
            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
            max_length=max_length,  # 填充 & 截断长度
            pad_to_max_length=True,  # 填充到 `max_length`
            return_attention_mask=True,  # 返回 attn. masks.
            return_tensors='pt',  # 返回 pytorch tensors 格式的数据
        )

        # 将编码后的文本添加到列表
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # 将列表转换为tensor
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

test_file_path = 'test.csv'

# 读取并预处理数据
test_labels, test_texts = read_ag_news_csv(test_file_path)
test_input_ids, test_attention_masks, test_labels_tensor = preprocess_data(tokenizer, test_texts, test_labels)

# 使用DataLoader
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=4)  # 可以调整batch_size


# 检查CUDA是否可用，并据此将模型移动到GPU或保持在CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def evaluate(model, dataloader):
    model.eval()  # 设置为评估模式

    predictions, true_labels = [], []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.extend(np.argmax(logits, axis=1))
        true_labels.extend(label_ids)

    return true_labels, predictions

true_labels, predictions = evaluate(model, test_dataloader)

# 计算性能指标
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")