import torch
from sklearn.metrics import f1_score


def binary_classification_results(prediction, ground_truth):
    """
    Computes results for binary classification
    Arguments:
    prediction (seq_length, batch_size, 1) - model predictions
    ground_truth (seq_length, batch_size, 1) - true labels
    Returns:
    tp_count, fp_count, fn_count, correct_count, total_count
    """
    true_positives_count = 0
    false_positives_count = 0
    false_negatives_count = 0
    correct_count = 0
    mask = ground_truth > -0.5
    total_count = torch.sum(mask)
    gt = torch.flatten(ground_truth[mask])
    pred = torch.flatten(prediction[mask])
    for j in range(total_count):
        if gt[j]:
            if pred[j]:
                true_positives_count += 1
                correct_count += 1
            else:
                false_negatives_count += 1
        else:
            if pred[j]:
                false_positives_count += 1
            else:
                correct_count += 1

    return true_positives_count, false_positives_count, false_negatives_count, correct_count, total_count


def binary_classification_metrics_by_counts(true_positives_count, false_positives_count, false_negatives_count,
                                            correct_count, total_count):
    """
    Computes metrics for binary classification
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    if true_positives_count + false_positives_count > 0:
        precision = true_positives_count / (true_positives_count + false_positives_count)
    else:
        precision = 1
    if true_positives_count + false_negatives_count > 0:
        recall = true_positives_count / (true_positives_count + false_negatives_count)
    else:
        recall = 1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    accuracy = correct_count / total_count

    return precision, recall, f1, accuracy


def binary_classification_metrics(prediction, ground_truth):
    """
    Computes metrics for binary classification
    Arguments:
    prediction (seq_length, batch_size, 1) - model predictions
    ground_truth (seq_length, batch_size, 1) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    true_positives_count, false_positives_count, false_negatives_count, correct_count, total_count = \
        binary_classification_results(prediction, ground_truth)
    return binary_classification_metrics_by_counts(true_positives_count, false_positives_count,
                                                   false_negatives_count, correct_count, total_count)


def calc_sklearn_f1_score(gts, preds, average):
    mask = gts > -0.5
    gts = gts[mask].flatten()
    preds = preds[mask].flatten()
    return f1_score(gts, preds, average=average)
