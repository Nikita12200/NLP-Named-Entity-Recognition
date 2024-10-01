from src.crf_pos_tag import load_brown_corpus, CRFPosTagger
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()
def evaluate_predictions(test_tagseqs, test_preds, all_labels,fold_idx):

    # Flatten test_tagseqs and preds while ensuring each tag is a string
    test_tags_flat = [tag for tagseq in test_tagseqs for tag in tagseq]
    preds_flat = [tag for pred in test_preds for tag in pred]

    accuracy = accuracy_score(test_tags_flat, preds_flat)
    precision, recall, f, _ = precision_recall_fscore_support(
        test_tags_flat, preds_flat, average="weighted",beta=1, zero_division=1
    )
    cm = confusion_matrix(test_tags_flat, preds_flat, labels=all_labels)
    class_report = classification_report(test_tags_flat, preds_flat, labels=all_labels, zero_division=1)
    print('after evaluation prediction')

    plt.figure(figsize=(11,11))
    sns.heatmap(cm,annot=True,fmt = 'd', cmap='Blues', cbar=True, 
            xticklabels=all_labels, 
            yticklabels=all_labels)
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f"heat_maps/confusion_matrix_heatmap_{fold_idx}.png")  # Save heatmap to a file
    plt.close()
    # plt.show()

    return accuracy, precision, recall, f, cm, class_report


def main(args: argparse.Namespace):
    wordseqs, tagseqs = load_brown_corpus(args.dataset_path)
    dataset = list(zip(wordseqs, tagseqs))

    # Extract all unique labels in the dataset
    all_labels = sorted({tag for tags in tagseqs for tag in tags})

    kf = KFold(args.k, random_state=42, shuffle=True)
    total_acc = 0
    total_prec =0
    total_recall=0
    total_f_score =0
    print(len(dataset))
    for fold_idx, (train_ids, test_ids) in enumerate(
        kf.split(np.zeros((len(dataset), 1)))
    ):
        print(f"Evaluating Fold {fold_idx + 1}/{args.k}")

        train_ds = [dataset[idx] for idx in train_ids]
        test_ds = [dataset[idx] for idx in test_ids]

        tagger = CRFPosTagger()
        tagger.train_crf(train_ds)

        test_tagseqs = [tagseq for _, tagseq in test_ds]
        test_preds = tagger.predict_crf([wordseq for wordseq, _ in test_ds], args.model_path)

        accuracy, precision, recall, f, cm, class_report = evaluate_predictions(
            test_tagseqs, test_preds, all_labels,fold_idx
        )
        total_acc+=accuracy
        total_prec+=precision
        total_recall+=recall
        total_f_score+=f
        print(f"Fold {fold_idx + 1} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F-Score: {f:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(class_report)
    avg_accuracy = total_acc / args.k
    avg_precision = total_prec / args.k
    avg_recall = total_recall / args.k
    avg_f2 = total_f_score / args.k

    print("\nAverage Results across all folds:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F2-Score: {avg_f2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-path", type=str, required=True)
    parser.add_argument("-k", "--k", type=int, default=5)
    parser.add_argument("-m", "--model-path", type=str)
    args = parser.parse_args()
    main(args)
