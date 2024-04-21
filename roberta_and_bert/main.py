import argparse
from datasets import DialogueDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16)
    args = parser.parse_args()

    train_path = './data/train/train.csv'
    test_path = './data/test/test.csv'
    batch_size = args.batch_size

    train_set = DialogueDataset(train_path, fields_used=["intent", "category"], is_val=False)
    train_loader = DataLoader(train_set, batch_size, shuffle=False)

    for data in train_loader:
        user1, text1, intent1, user2, text2, intent2, category, label = data
