import argparse
import json
import pickle
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
            raise ValueError("ValueError in string_to_dict")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def save_nli_data_csv(ins_list:list,path:str):
    assert path.endswith(".csv"), "should be a csv file."
    
    with open(path,"w") as csvfile: 
        writer = csv.writer(csvfile)

        writer.writerow(["sentence1","sentence2","label"])
        writer.writerows(ins_list)

def save_t2t_data_csv(ins_list:list,path:str):
    assert path.endswith(".csv"), "should be a csv file."
    
    with open(path,"w") as csvfile: 
        writer = csv.writer(csvfile)

        writer.writerow(["text_1","text_2","label"])
        writer.writerows(ins_list)

def utt2text(utt:dict):
    # TODO: now only using the `user` and `text` field in the original data
    return f"{utt['user']}: {utt['text']}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path",type=str,default='./data')
    parser.add_argument("--target_path",type=str,default="./data")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--split",action="store_true",default=False)
    parser.add_argument("--shuffle",type=bool,default=True)

    args, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError(unparsed)

    label2id = {} 

    target_path = args.target_path
    source_path = args.source_path
    seed = args.seed
    
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(target_path,exist_ok=True)

    # open csv file
    with open(os.path.join(source_path,"train" ,"train.csv"),"r") as f:
        reader = csv.reader(f)
        train_data = list(reader)
    with open(os.path.join(source_path, "test" ,"test.csv"),"r") as f:
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
        assert label in ["0","1"], "assert label should be 0 or 1. but got {}".format(label)
        text_1 = utt2text(utt_1)
        text_2 = utt2text(utt_2)
        train_samples.append((text_1,text_2,label))
    
    if args.shuffle:
        combined = list(zip(train_samples, train_data))
        # Shuffle the combined list
        random.shuffle(combined)
        # Unzip the result back into separate lists
        train_samples, train_data = zip(*combined)
    
    # split 0.1 for eval
    if args.split:
        train_samples, eval_samples = train_samples[:-int(len(train_samples)*0.1)], train_samples[-int(len(train_samples)*0.1):]
        train_data, eval_data = train_data[:-int(len(train_data)*0.1)], train_data[-int(len(train_data)*0.1):]
    else:
        train_samples, eval_samples = train_samples, []
        train_data, eval_data = train_data, []
    
    # for test data
    test_samples = []
    for i, ins in tqdm(enumerate(test_data)):
        label, cate, utt_1, utt_2 = ins[0], ins[1], ins[2], ins[3]
        utt_1 = string_to_dict(utt_1)
        utt_2 = string_to_dict(utt_2)
        label = label.strip()
        assert label in ["0","1"], "assert label should be 0 or 1. but got {}".format(label)
        text_1 = utt2text(utt_1)
        text_2 = utt2text(utt_2)
        test_samples.append((text_1,text_2,label))

    save_nli_data_csv(train_samples,os.path.join(target_path,"train.csv"))
    save_nli_data_csv(test_samples,os.path.join(target_path,"test.csv"))
    save_nli_data_csv(eval_samples,os.path.join(target_path,"eval.csv"))
        
    label2id = {"0":0,"1":1}
    
    with open(os.path.join(target_path,"categories2id.json"),"w") as f:
        json.dump(label2id,f)


if __name__ == "__main__":
    main()