import csv

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, GPT2Model, GPT2Config, GPT2Tokenizer, BertTokenizer
import torch


class BertGPT2Model(torch.nn.Module):
    def __init__(self, bert_model_path, gpt2_model_path):
        super().__init__()
        # 加载本地BERT模型
        self.bert = BertModel.from_pretrained(bert_model_path)
        # 加载本地GPT2配置并初始化模型
        self.gpt2_config = GPT2Config.from_pretrained(gpt2_model_path)
        self.gpt2 = GPT2Model.from_pretrained(gpt2_model_path, config=self.gpt2_config)

        # 调整BERT输出特征的维度以匹配GPT2的嵌入维度
        self.bert_to_gpt2 = torch.nn.Linear(self.bert.config.hidden_size, self.gpt2.config.n_embd)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # 从BERT模型获取特征
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_output = outputs.last_hidden_state[:, 0, :]  # 取CLS标记的输出作为特征

        # 将BERT的输出特征维度调整为GPT2的嵌入维度
        bert_features = self.bert_to_gpt2(bert_output)

        # 这里简化处理，实际使用时可能需要更复杂的逻辑来整合特征
        gpt2_output = self.gpt2(inputs_embeds=bert_features.unsqueeze(0))

        return gpt2_output


# 示例初始化模型，使用本地模型路径
model = BertGPT2Model('myBertModel', 'myGptModel')
tokenizer = BertTokenizer.from_pretrained("myBertModel")

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
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,  # 输入文本
            add_special_tokens=True,  # 添加特殊令牌
            max_length=max_length,  # 指定最大长度
            pad_to_max_length=True,  # 执行填充
            return_attention_mask=True,  # 返回注意力掩码
            return_tensors='pt',  # 返回PyTorch张量
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 执行截断
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
test_dataloader = DataLoader(test_dataset, batch_size=1)  # 可以调整batch_size

# 转换为评估模式
model.eval()

# 初始化用于保存预测和真实标签的列表
true_labels = []
pred_labels = []

# 检查CUDA是否可用，并据此将模型移动到GPU或保持在CPU
device = 'cpu'
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
