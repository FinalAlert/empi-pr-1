import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

SURNAME = "dedesh"


df = pd.read_csv("data_metrics.csv")
print(df.head())

thresh = 0.5
df["predicted_RF"] = (df.model_RF >= thresh).astype(int)
df["predicted_LR"] = (df.model_LR >= thresh).astype(int)
print(df.head())


def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return int(np.sum((y_true == 1) & (y_pred == 1)))


def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return int(np.sum((y_true == 1) & (y_pred == 0)))


def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return int(np.sum((y_true == 0) & (y_pred == 1)))


def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return int(np.sum((y_true == 0) & (y_pred == 0)))


print("TP:", find_TP(df.actual_label.values, df.predicted_RF.values))
print("FN:", find_FN(df.actual_label.values, df.predicted_RF.values))
print("FP:", find_FP(df.actual_label.values, df.predicted_RF.values))
print("TN:", find_TN(df.actual_label.values, df.predicted_RF.values))


def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN


def dedesh_confusion_matrix(y_true, y_pred):
    """Матриця помилок у форматі sklearn: [[TN, FP], [FN, TP]]."""
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


# alias, щоб збігалося з текстом методички після заміни прізвища
my_confusion_matrix = dedesh_confusion_matrix

print("sklearn RF:\n", confusion_matrix(df.actual_label.values, df.predicted_RF.values))
print(f"{SURNAME}_confusion_matrix RF:\n", dedesh_confusion_matrix(df.actual_label.values, df.predicted_RF.values))

assert np.array_equal(
    dedesh_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
    confusion_matrix(df.actual_label.values, df.predicted_RF.values),
), "dedesh_confusion_matrix() is not correct for RF"
assert np.array_equal(
    dedesh_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
    confusion_matrix(df.actual_label.values, df.predicted_LR.values),
), "dedesh_confusion_matrix() is not correct for LR"
print("confusion_matrix asserts: OK")


def dedesh_accuracy_score(y_true, y_pred):
    # (TP + TN) / (TP + TN + FP + FN)
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)


my_accuracy_score = dedesh_accuracy_score

assert dedesh_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(
    df.actual_label.values, df.predicted_RF.values
), "dedesh_accuracy_score failed on RF"
assert dedesh_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(
    df.actual_label.values, df.predicted_LR.values
), "dedesh_accuracy_score failed on LR"
print("Accuracy RF: %.3f" % dedesh_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print("Accuracy LR: %.3f" % dedesh_accuracy_score(df.actual_label.values, df.predicted_LR.values))


def dedesh_recall_score(y_true, y_pred):
    # TP / (TP + FN)
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)


my_recall_score = dedesh_recall_score

assert dedesh_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(
    df.actual_label.values, df.predicted_RF.values
), "dedesh_recall_score failed on RF"
assert dedesh_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(
    df.actual_label.values, df.predicted_LR.values
), "dedesh_recall_score failed on LR"
print("Recall RF: %.3f" % dedesh_recall_score(df.actual_label.values, df.predicted_RF.values))
print("Recall LR: %.3f" % dedesh_recall_score(df.actual_label.values, df.predicted_LR.values))


def dedesh_precision_score(y_true, y_pred):
    # TP / (TP + FP)
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)


my_precision_score = dedesh_precision_score

assert dedesh_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(
    df.actual_label.values, df.predicted_RF.values
), "dedesh_precision_score failed on RF"
assert dedesh_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(
    df.actual_label.values, df.predicted_LR.values
), "dedesh_precision_score failed on LR"
print("Precision RF: %.3f" % dedesh_precision_score(df.actual_label.values, df.predicted_RF.values))
print("Precision LR: %.3f" % dedesh_precision_score(df.actual_label.values, df.predicted_LR.values))


def dedesh_f1_score(y_true, y_pred):
    # 2 * precision * recall / (precision + recall)
    recall = dedesh_recall_score(y_true, y_pred)
    precision = dedesh_precision_score(y_true, y_pred)
    return 2 * precision * recall / (precision + recall)


my_f1_score = dedesh_f1_score

assert np.isclose(
    dedesh_f1_score(df.actual_label.values, df.predicted_RF.values),
    f1_score(df.actual_label.values, df.predicted_RF.values),
), "dedesh_f1_score failed on RF"
assert np.isclose(
    dedesh_f1_score(df.actual_label.values, df.predicted_LR.values),
    f1_score(df.actual_label.values, df.predicted_LR.values),
), "dedesh_f1_score failed on LR"
print("F1 RF: %.3f" % dedesh_f1_score(df.actual_label.values, df.predicted_RF.values))
print("F1 LR: %.3f" % dedesh_f1_score(df.actual_label.values, df.predicted_LR.values))

print("\nscores with threshold = 0.5")
print("Accuracy RF: %.3f" % dedesh_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print("Recall RF: %.3f" % dedesh_recall_score(df.actual_label.values, df.predicted_RF.values))
print("Precision RF: %.3f" % dedesh_precision_score(df.actual_label.values, df.predicted_RF.values))
print("F1 RF: %.3f" % dedesh_f1_score(df.actual_label.values, df.predicted_RF.values))

print("\nscores with threshold = 0.25")
pred_rf_025 = (df.model_RF >= 0.25).astype(int).values
print("Accuracy RF: %.3f" % dedesh_accuracy_score(df.actual_label.values, pred_rf_025))
print("Recall RF: %.3f" % dedesh_recall_score(df.actual_label.values, pred_rf_025))
print("Precision RF: %.3f" % dedesh_precision_score(df.actual_label.values, pred_rf_025))
print("F1 RF: %.3f" % dedesh_f1_score(df.actual_label.values, pred_rf_025))

fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print("AUC RF: %.3f" % auc_RF)
print("AUC LR: %.3f" % auc_LR)

plt.figure(figsize=(7, 6))
plt.plot(fpr_RF, tpr_RF, "r-", label="RF AUC: %.3f" % auc_RF)
plt.plot(fpr_LR, tpr_LR, "b-", label="LR AUC: %.3f" % auc_LR)
plt.plot([0, 1], [0, 1], "k-", label="random")
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], "g-", label="perfect")
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves: Random Forest vs Logistic Regression")
plt.tight_layout()
plt.savefig("roc_rf_lr.png", dpi=120)
print("ROC saved to roc_rf_lr.png")