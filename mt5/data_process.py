import argparse
import os
import csv
import random
import numpy as np
from tqdm import tqdm
import ast


def string_to_dict(dict_string):
    try:
        # Convert the string representation of a dict to an actual dictionary
        result = ast.literal_eval(dict_string)
        if isinstance(result, dict):
            return result
        else:
            raise ValueError("The string does not represent a dictionary")
    except Exception as e:
        # Handle specific error cases or re-raise the error
        print(f"An error occurred: {e}")
        # Optionally, you can re-raise the exception after logging
        raise

def save_nli_data_csv(ins_list:list,path:str):
    assert path.endswith(".csv"), "should be a csv file."

    with open(path,"w") as csvfile:
        writer = csv.writer(csvfile)

        # columns name
        writer.writerow(["sentence1", "sentence2", "label"])
        # all instances
        writer.writerows(ins_list)

def save_t2t_data_csv(ins_list:list,path:str):
    assert path.endswith(".csv"), "should be a csv file."

    with open(path,"w") as csvfile:
        writer = csv.writer(csvfile)
        # columns name
        writer.writerow(["text_1", "text_2", "label"])
        # all instances
        writer.writerows(ins_list)

def utt2text(utt:dict):
    # TODO: now only using the `user` and `text` field in the original data
    return f"{utt['user']}: {utt['text']}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="./data")
    parser.add_argument("--target_path", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", type=bool, default=True)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)

    target_path = args.target_path
    source_path = args.source_path
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(target_path,exist_ok=True)

    # open csv file
    with open(os.path.join(source_path, "train", "train.csv"), "r") as f:
        reader = csv.reader(f)
        train_data = list(reader)
    with open(os.path.join(source_path, "test", "test.csv"), "r") as f:
        reader = csv.reader(f)
        test_data = list(reader)

    # remove head
    train_data = train_data[1:]
    test_data = test_data[1:]

    # for training/eval data
    train_samples = []
    for i, ins in tqdm(enumerate(train_data)):
        label, cate, utt_1, utt_2 = ins[0], ins[1], ins[2], ins[3]
        # convert the utt (string) to the dict
        utt_1 = string_to_dict(utt_1)
        utt_2 = string_to_dict(utt_2)
        label = label.strip()
        assert label in ["0", "1"], "assert label should be 0 or 1. but got {}".format(label)
        text_1 = utt2text(utt_1)
        text_2 = utt2text(utt_2)
        train_samples.append((text_1,text_2,label))

    if args.shuffle:
        combined = list(zip(train_samples, train_data))
        # Shuffle the combined list
        random.shuffle(combined)
        # Unzip the result back into separate lists
        train_samples, train_data = zip(*combined)

    # for test data
    test_samples = []
    for i, ins in tqdm(enumerate(test_data)):
        label, cate, utt_1, utt_2 = ins[0], ins[1], ins[2], ins[3]
        utt_1 = string_to_dict(utt_1)
        utt_2 = string_to_dict(utt_2)
        label = label.strip()
        assert label in ["0", "1"], "assert label should be 0 or 1. but got {}".format(label)
        text_1 = utt2text(utt_1)
        text_2 = utt2text(utt_2)
        test_samples.append((text_1,text_2,label))

    save_nli_data_csv(train_samples, os.path.join(target_path, "train.csv"))
    save_nli_data_csv(test_samples, os.path.join(target_path, "test.csv"))

    print("train samples: {}, test samples: {}".format(len(train_samples), len(test_samples)))
    print("data saved in {}".format(target_path))


if __name__ == "__main__":
    main()
