import argparse
from datasets import DialogueDataset
from torch.utils.data import DataLoader
from encoder import BertEncoder, RobertaEncoder
import torch.nn as nn
from transformers import AdamW
from metrics import calculate
import torch
import os


class DialogueClassifier(nn.Module):
    def __init__(self, bert_model, roberta_model):
        super(DialogueClassifier, self).__init__()
        self.bert = bert_model
        self.roberta = roberta_model
        self.classifier = nn.Linear(768 * 2, 2)

    def forward(self, utterance1, utterance2):
        bert_embeddings = self.bert.encode(utterance1)
        roberta_embeddings = self.roberta.encode(utterance2)
        combined_embeddings = torch.cat((bert_embeddings, roberta_embeddings), dim=1)
        logits = self.classifier(combined_embeddings)
        return logits


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epoch", type=int, default=5)
    args = parser.parse_args()

    train_path = './data/train/train.csv'
    test_path = './data/test/test.csv'
    batch_size = args.batch_size
    lr = args.lr
    if not os.path.isdir("./result/"):
        os.makedirs("./result/")

    if not os.path.isdir("./model/"):
        os.makedirs("./model/")

    log_file_path = os.path.join("./result/", f'lr_{args.lr}_bs_{args.batch_size}_epoch{args.epoch}.txt')
    with open(log_file_path, 'w') as log_file:
        pass

    train_set = DialogueDataset(train_path, fields_used=["intent", "category"], is_val=False)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    test_set = DialogueDataset(test_path, fields_used=["intent", "category"], is_val=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    bert_encoder = BertEncoder(pretrained_model_name='bert-base-uncased', device=device)
    roberta_encoder = RobertaEncoder(pretrained_model_name='roberta-base', device=device)
    model = DialogueClassifier(bert_encoder, roberta_encoder).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(args.epoch):
        for data in train_loader:
            user1, text1, intent1, user2, text2, intent2, category, label = data
            label = label.to(torch.long).to(device)
            optimizer.zero_grad()
            logits = model(text1, text2)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}\n")

    torch.save(model.state_dict(), f'./model/model_lr_{args.lr}_bs_{args.batch_size}_epoch_{args.epoch}.pth')
    model.load_state_dict(torch.load(f'./model/model_lr_{args.lr}_bs_{args.batch_size}_epoch_{args.epoch}.pth'))
    model.eval()
    predictions, ground_truths = [], []
    with torch.no_grad():
        for data in test_loader:
            user1, text1, intent1, user2, text2, intent2, category, label = data
            logits = model(text1, text2)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            ground_truths.extend(label.cpu().numpy())

    metrics = calculate(ground_truths, predictions)
    print(metrics)
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{metrics}')
