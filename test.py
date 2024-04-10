import csv
import torch
from tqdm.auto import tqdm
from torch import nn
from transformers import BertModel, GPT2Model, AutoConfig, BertTokenizer
from transformers import BertModel, GPT2Model, AutoConfig
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, RandomSampler


class BertGPT2Classifier(nn.Module):
    def __init__(self, num_labels):
        super(BertGPT2Classifier, self).__init__()
        self.bert = BertModel.from_pretrained("pretrainedmodels/myBERT")
        self.gpt2 = GPT2Model.from_pretrained("pretrainedmodels/myGPT2")
        self.dropout = nn.Dropout(0.1)
        # 由于GPT2可能使用不同的隐藏层大小，我们基于GPT2的配置进行调整
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        # 使用BERT作为编码器
        encoder_outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = encoder_outputs[0]

        # 直接将BERT的输出作为输入
        gpt2_output = self.gpt2(inputs_embeds=sequence_output)

        # 从GPT-2输出中获取用于分类的特征
        gpt2_feature = gpt2_output.last_hidden_state[:, 0]

        gpt2_feature = self.dropout(gpt2_feature)
        logits = self.classifier(gpt2_feature)

        return logits


model = BertGPT2Classifier(num_labels=4)
# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained('pretrainedmodels/myBERT/')


def train(model, dataloader, epochs=5):
    # 检查是否有可用的GPU，如果有，则使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for training.")

    model.to(device)  # 将模型移到正确的设备上
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()  # 确保模型处于训练模式
        total_loss = 0

        # 使用tqdm进度条封装你的数据加载器
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()
            logits = model(b_input_ids, attention_mask=b_input_mask)
            loss = loss_fn(logits, b_labels)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # 更新进度条的显示信息
            progress_bar.set_postfix({'loss': loss.item()})

        # 每个epoch结束时打印平均损失
        print(f"Average loss: {total_loss / len(dataloader)}\n")

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
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 使用eos_token作为pad_token
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
            pad_to_max_length=True,  # 确保填充到max_length的长度
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

# 读取并预处理数据
train_labels, train_texts = read_ag_news_csv(train_file_path)
train_input_ids, train_attention_masks, train_labels = preprocess_data(tokenizer, train_texts, train_labels)

# 创建TensorDataset和DataLoader
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train(model, train_dataloader)