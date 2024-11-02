import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, average_precision_score
)

def compute_metrics(true_labels_file, predicted_labels_file):
    true_df = pd.read_csv(true_labels_file, sep='\t')  
    pred_df = pd.read_csv(predicted_labels_file, sep='\t')  

    labels = true_df['label'].values
    preds = pred_df['label'].values  

    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')

    class_report = classification_report(y_true=labels, y_pred=preds, target_names=['Class 0', 'Class 1'], output_dict=True)

    conf_matrix = confusion_matrix(y_true=labels, y_pred=preds)

    map_weighted = average_precision_score(y_true=labels, y_score=preds, average='weighted')
    map_macro = average_precision_score(y_true=labels, y_score=preds, average='macro')

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "map_weighted": map_weighted,
        "map_macro": map_macro,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix
    }

    return metrics

true_labels_file = 'thesis-codebase/data/cw_diabetes_test.tsv'
predicted_labels_file = 'thesis-codebase/data/results/deberta-base_clef,nlp4if,cw_diabetes,policlaim_42_1e-3_16_adapter_fusion/cw_diabetes/test_result.tsv'

metrics = compute_metrics(true_labels_file, predicted_labels_file)

print("Accuracy:", metrics["accuracy"])
print("Precision (Macro):", metrics["precision_macro"])
print("Recall (Macro):", metrics["recall_macro"])
print("F1-score (Macro):", metrics["f1_macro"])
print("Mean Average Precision (Weighted):", metrics["map_weighted"])
print("Mean Average Precision (Macro):", metrics["map_macro"])

print("\nClassification Report:")
for label, report in metrics["classification_report"].items():
    if isinstance(report, dict):
        print(f"{label}:")
        for metric, value in report.items():
            print(f"  {metric}: {value}")
    else:
        print(f"{label}: {report}")

print("\nConfusion Matrix:")
print(metrics["confusion_matrix"])
