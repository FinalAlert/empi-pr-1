import numpy as np
import matplotlib.pyplot as plt


def visualize_classifier(classifier, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    step_size = 0.01

    x_values, y_values = np.meshgrid(
        np.arange(x_min, x_max, step_size),
        np.arange(y_min, y_max, step_size)
    )

    mesh_output = classifier.predict(
        np.c_[x_values.ravel(), y_values.ravel()]
    )

    mesh_output = mesh_output.reshape(x_values.shape)

    plt.figure()

    plt.pcolormesh(
        x_values,
        y_values,
        mesh_output,
        shading='auto',
        cmap=plt.cm.Paired
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=80,
        edgecolors='black',
        cmap=plt.cm.Paired
    )

    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Logistic Regression Classifier')

    plt.show()