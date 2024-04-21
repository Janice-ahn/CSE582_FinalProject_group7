import json
from torch.utils.data import Dataset
import pandas as pd
import re


class DialogueDataset(Dataset):
    def _process_quotes(self, text):
        if '"' in text:
            return text.replace('"', "'")
        else:
            return text.replace("'", '"')

    def _fix_double_quotes(self, text):
        x = re.search(r'"text": (.+), "intent":', text)
        new_text = text
        if x[1].count('"') > 2:
            new_text = new_text.replace(x[1], x[1].replace('""', '"'))
            x = re.search(r'"text": (.+), "intent":', new_text)
            if x[1].count('"') > 2:
                new_text = new_text.replace(x[1][1:-1], x[1][1:-1].replace('"', ''))
        return new_text

    def __init__(self,
                 data_path: str,
                 fields_used: list,
                 is_val: bool = False):

        super().__init__()
        self.data = pd.read_csv(data_path)

        self.data["utterance1"] = self.data["utterance1"].apply(lambda x: re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1\2", x))
        self.data["utterance2"] = self.data["utterance2"].apply(lambda x: re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1\2", x))

        self.data["utterance1"] = self.data["utterance1"].apply(self._process_quotes)
        self.data["utterance2"] = self.data["utterance2"].apply(self._process_quotes)

        self.data["utterance1"] = self.data["utterance1"].apply(lambda x: x.replace("'", '"'))
        self.data["utterance2"] = self.data["utterance2"].apply(lambda x: x.replace("'", '"'))

        self.data["utterance1"] = self.data["utterance1"].apply(self._fix_double_quotes)
        self.data["utterance2"] = self.data["utterance2"].apply(self._fix_double_quotes)

        self.data["utterance1"] = self.data["utterance1"].apply(lambda x: x.replace("\\", ""))
        self.data["utterance2"] = self.data["utterance2"].apply(lambda x: x.replace("\\", ""))

        category_used = False
        if "category" in fields_used:
            category_used = True

        intent_used = False
        if "intent" in fields_used:
            intent_used = True
        self.is_val = is_val
        self.intent_used = intent_used
        self.category_used = category_used

    def __len__(self):
        return len(self.data)

    def _get_intent_and_category(self, utt1, utt2, category):
        intent1 = ""
        intent2 = ""
        cat = ""
        if self.intent_used:
            intent1 = utt1["intent"]
            intent2 = utt2["intent"]
        if self.category_used:
            cat = category
        return intent1, intent2, cat

    def __getitem__(self, index):
        row = self.data.iloc[index]
        utt1 = json.loads(row["utterance1"])
        utt2 = json.loads(row["utterance2"])
        label = row["label"]
        intent1, intent2, cat = self._get_intent_and_category(utt1, utt2, row["category"])
        user1 = utt1["user"]
        user2 = utt2["user"]
        text1 = utt1["text"]
        text2 = utt2["text"]

        return user1, text1, intent1, user2, text2, intent2, cat, label
