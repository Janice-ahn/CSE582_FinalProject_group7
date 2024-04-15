import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from datasets import load_dataset

import torch
import evaluate
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from model import (
    SequenceClassificationTrainer,
    SequenceClassificationConfig,
    SequenceClassificationEncoder
)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    label_column_name: Optional[str] = field(
        default="label",
        metadata={
            "help": "The name of the label column in the datasets."
        }
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    classifier_dropout: float = field(
        default=0.2,
        metadata={"help": "dropout rate for classification head."}
    )


@dataclass
class SelfTrainingArguments(TrainingArguments):
    learning_rate_cls: Optional[float] = field(
        default=1e-3,
        metadata={
            "help": "The learning rate for projection head."
        }
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, SelfTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Loading a dataset from your local csv files
    data_files = {
        "train": data_args.train_file,
        "validation": data_args.test_file,
        "test": data_args.test_file
    }

    raw_datasets = load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=None,
    )

    # Labels
    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    label_list = raw_datasets["train"].unique(data_args.label_column_name)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    config = SequenceClassificationConfig.from_pretrained(
        model_args.model_name_or_path,
        finetuning_task=None,
        cache_dir=model_args.cache_dir,
        revision="main",
        use_auth_token=None,
    )
    config.get_additional_args(num_labels=num_labels, classifier_dropout=model_args.classifier_dropout)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision="main",
        use_auth_token=None
    )

    model = SequenceClassificationEncoder.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision="main",
        use_auth_token=None
    )

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the raw_datasets
    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [name for name in raw_datasets["train"].column_names if name != data_args.label_column_name]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=max_seq_length, truncation=True)

        result["label"] = [l for l in examples[data_args.label_column_name]]
        # print("result", result["label"][:10])

        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset"
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]

    # Get the metric function
    acc_metric = evaluate.load("accuracy")
    f_metric = evaluate.load("f1")
    r_metric = evaluate.load('recall')
    p_metric = evaluate.load('precision')

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)

        result = dict()
        result.update(acc_metric.compute(predictions=preds, references=p.label_ids))
        result.update(f_metric.compute(predictions=preds, references=p.label_ids, average="macro"))
        result.update(p_metric.compute(predictions=preds, references=p.label_ids, average="macro"))
        result.update(r_metric.compute(predictions=preds, references=p.label_ids, average="macro"))

        result = {
            k: round(v * 100, 4) for k, v in result.items()
        }

        return result

    # Initialize trainer
    trainer = SequenceClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    trainer.init_hyper(lr=training_args.learning_rate, lr_cls=training_args.learning_rate_cls)

    # train
    if training_args.do_train:
        train_result = trainer.train()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # prediction
    if training_args.do_predict:
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = predict_results.predictions

        ## save the probability
        prob = torch.nn.functional.softmax(torch.tensor(predictions), dim=1).tolist()
        output_predict_prob_file = os.path.join(training_args.output_dir, f"predict_prob.txt")

        with open(output_predict_prob_file, "w") as prob_writer:
            prob_writer.write("index\tFalse\tTrue\n")
            for index, p in enumerate(prob):
                prob_writer.write(f"{index}\t{p[0]}\t{p[1]}\n")

        # add metrics on test set
        metrics_test = predict_results.metrics
        predictions = np.argmax(predictions, axis=1)

        trainer.log_metrics("predict", metrics_test)
        trainer.save_metrics("predict", metrics_test)

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")

if __name__ == "__main__":
    main()
