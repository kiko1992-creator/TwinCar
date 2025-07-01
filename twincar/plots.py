import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_metrics(train_hist, val_hist, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    ax[0].plot(train_hist['loss'], label='Train Loss')
    ax[0].plot(val_hist['loss'], label='Val Loss')
    ax[0].set_title('Loss')
    ax[0].legend()
    ax[1].plot(train_hist['acc'], label='Train Acc')
    ax[1].plot(val_hist['acc'], label='Val Acc')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    plt.savefig(out_path)
    plt.close()

def plot_confusion(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(out_path)
    plt.close()
