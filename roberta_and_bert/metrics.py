from sklearn.metrics import f1_score, precision_score, recall_score


def calculate(ground_truths, predictions):
    f1_1 = f1_score(ground_truths, predictions, average='binary', pos_label=1)
    f1_0 = f1_score(ground_truths, predictions, average='binary', pos_label=0)
    f1_macro = f1_score(ground_truths, predictions, average='macro')

    prec_1 = precision_score(ground_truths, predictions, average='binary', pos_label=1)
    prec_0 = precision_score(ground_truths, predictions, average='binary', pos_label=0)
    prec_macro = precision_score(ground_truths, predictions, average='macro')

    rec_1 = recall_score(ground_truths, predictions, average='binary', pos_label=1)
    rec_0 = recall_score(ground_truths, predictions, average='binary', pos_label=0)
    rec_macro = recall_score(ground_truths, predictions, average='macro')

    return {"Label 0": {"F1": f1_0, "Precision": prec_0, "Recall": rec_0},
            "Label 1": {"F1": f1_1, "Precision": prec_1, "Recall": rec_1},
            "Overall": {"F1": f1_macro, "Precision": prec_macro, "Recall": rec_macro}}
