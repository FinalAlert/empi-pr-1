import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

input_file = "data_multivar_nb.txt"
data = np.loadtxt(input_file, delimiter=",")
X, y = data[:, :-1], data[:, -1]


def visualize_classifier(classifier, X, y, title, filename):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    mesh_step_size = 0.02
    x_vals, y_vals = np.meshgrid(
        np.arange(min_x, max_x, mesh_step_size),
        np.arange(min_y, max_y, mesh_step_size),
    )
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    plt.figure(figsize=(7, 6))
    plt.title(title)
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray, shading="auto")
    plt.scatter(
        X[:, 0], X[:, 1], c=y, s=50, edgecolors="black", linewidth=1, cmap=plt.cm.Paired
    )
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    print("saved", filename)
    plt.show()


def report(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n===== {name} =====")
    print("Accuracy:", round(100 * accuracy_score(y_test, y_pred), 2), "%")
    print(
        "Precision:",
        round(100 * precision_score(y_test, y_pred, average="weighted", zero_division=0), 2),
        "%",
    )
    print(
        "Recall:",
        round(100 * recall_score(y_test, y_pred, average="weighted", zero_division=0), 2),
        "%",
    )
    print(
        "F1:",
        round(100 * f1_score(y_test, y_pred, average="weighted", zero_division=0), 2),
        "%",
    )
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    num_folds = 3
    for scoring, label in (
        ("accuracy", "Accuracy"),
        ("precision_weighted", "Precision"),
        ("recall_weighted", "Recall"),
        ("f1_weighted", "F1"),
    ):
        values = cross_val_score(model, X, y, scoring=scoring, cv=num_folds)
        print(f"CV {label}: {round(100 * values.mean(), 2)}%")
    return model


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3
)

svm_model = report(
    "SVM (RBF)",
    SVC(kernel="rbf", C=1.0, gamma="scale"),
    X_train,
    X_test,
    y_train,
    y_test,
)
nb_model = report(
    "Gaussian Naive Bayes",
    GaussianNB(),
    X_train,
    X_test,
    y_train,
    y_test,
)

visualize_classifier(
    svm_model, X_test, y_test, "SVM classifier", "svm_classifier.png"
)
visualize_classifier(
    nb_model, X_test, y_test, "Naive Bayes classifier", "nb_classifier.png"
)