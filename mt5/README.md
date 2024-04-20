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
sh main.sh [GPU] [batch_size] [model_name] [learning_rate]

sh main.sh 7 32 google/mt5-small 5e-4
sh main.sh 7 32 google/mt5-base 5e-4
sh main.sh 7 16 google/mt5-large 5e-4
```


## Results

| Model          | ACC      | F1       | Precision | Recall   |
|----------------|----------|----------|-----------|----------|
| mt5-small      | 73.72    | 69.05    | 69.07     | 69.04    |
| mt5-small (S1) | 79.15    | 74.74    | 75.60     | 74.08    |
| mt5-small (S2) | 75.58    | 71.64    | 71.38     | 71.94    |
| mt5-base       | 83.02    | 80.04    | 80.01     | 80.06    |
| mt5-base (S1)  | 77.75    | 74.09    | 73.86     | 74.35    |
| mt5-base (S2)  | 73.95    | 68.42    | 69.10     | 67.93    |
| mt5-large      | 78.76    | 75.04    | 75.00     | 75.08    |
| mt5-large (S1) | 79.38    | 75.34    | 75.79     | 74.96    |
| mt5-large (S2) | 76.67    | 71.03    | 72.61     | 70.10    |
