import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


class ModelMetrics:
    def __init__(self, model, generator):
        self.model = model
        self.generator = generator

    def evaluate(self):
        # Predictions
        y_pred_probs = self.model.predict(self.generator)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = self.generator.classes

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification Report
        report = classification_report(
            y_true,
            y_pred,
            target_names=list(self.generator.class_indices.keys())
        )

        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)

        # Plot Confusion Matrix
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()

        # ROC Curve (for binary classification)
        if y_pred_probs.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.savefig("roc_curve.png")
            plt.close()

        return cm, report