# Dialogue Segment Classification

## Overview

The project consists of several components:

1. **Data Preprocessing**: The `datasets.py` module preprocesses the dialogue data, including tokenization, formatting,
   and loading into a suitable format for training.

2. **Model Architecture**: The `encoder.py` module defines encoder classes (`BertEncoder` and `RobertaEncoder`) for
   encoding dialogue segments using pre-trained BERT and RoBERTa models respectively.

3. **Model Training**: The `train.py` script orchestrates the training process. It loads the dataset, initializes the
   model, trains the model on the training set, and evaluates its performance on the test set.

4. **Evaluation Metrics**: The `metrics.py` module provides functions to calculate evaluation metrics such as F1-score,
   precision, and recall for assessing the model's performance.

## Requirements

- Python 3
- PyTorch
- Transformers library (for BERT and RoBERTa models)
- scikit-learn

## Usage

1. Clone the repository:

```bash
git clone https://github.com/Janice-ahn/CSE582_FinalProject_group7.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
    - Place your training and test CSV files in the `data/train/` and `data/test/` directories respectively.

4. Run the training script:

```bash
python train.py --batch_size 1024 --lr 0.001 --epoch 5
```

Replace `batch_size`, `lr`, and `epoch` with your desired values.

5. After training, the model will be saved in the `model/` directory, and evaluation metrics will be logged in
   the `result/` directory.
