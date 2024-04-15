# CSE 582

[MT5](https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/mt5) is used.


## Environment

Create an environment with:

- Python 3.11.4
- PyTorch 2.0.1

Then install dependencies:

```bash
pip install -r requirements.txt
```


## Datasets

Download the [dataset files](https://drive.google.com/drive/folders/1RAWWGTI7ciFkQfl3P9TSlC8Wm-seZYrN) and put them under the `./data` folder. The file structure should be like this:

```
├── data
│   ├── other-dialogue-transcripts
│   ├── test
│   ├── train
│   └── videos
├── main.py
...
```

Then, preprocess the data:

```bash
python data_process.py
```


##  Train

```bash
sh run.sh [GPU] [batch_size] [model_name] [learning_rate]

sh run.sh 7 32 google/mt5-small 5e-4
sh run.sh 7 32 google/mt5-base 5e-4
sh run.sh 7 16 google/mt5-large 5e-4
```


## Results

| Model     | ACC      | F1       | Precision | Recall   |
|-----------|----------|----------|-----------|----------|
| mt5-small | 73.72    | 69.05    | 69.07     | 69.04    |
| mt5-base  | 83.02    | 80.04    | 80.01     | 80.06    |
| mt5-large | 78.76    | 75.04    | 75.00     | 75.08    |
