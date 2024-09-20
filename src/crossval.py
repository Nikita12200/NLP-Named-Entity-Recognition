from src.pos_tagger import load_brown_corpus, HMMPosTagger
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import argparse
import numpy as np


def evaluate_predictions(test_tagseqs, test_preds):
    test_tags_flat = [tag for tagseq in test_tagseqs for tag in tagseq.lower().split()]
    preds_flat = [tag for pred in test_preds for tag in pred.lower().split()]

    accuracy = accuracy_score(test_tags_flat, preds_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_tags_flat, preds_flat, average="weighted"
    )
    cm = confusion_matrix(test_tags_flat, preds_flat)
    class_report = classification_report(test_tags_flat, preds_flat)
    return accuracy, precision, recall, f1, cm, class_report


def main(args: argparse.Namespace):
    wordseqs, tagseqs = load_brown_corpus(args.dataset_path)
    dataset = list(zip(wordseqs, tagseqs))

    kf = KFold(args.k, random_state=42, shuffle=True)

    for fold_idx, (train_ids, test_ids) in enumerate(
        kf.split(np.zeros((len(dataset), 1)))
    ):
        print(f"Evaluating Fold {fold_idx + 1}/{args.k}")

        train_ds = [dataset[idx] for idx in train_ids]
        test_ds = [dataset[idx] for idx in test_ids]

        tagger = HMMPosTagger()
        tagger.train(train_ds, smoothing=args.smoothing_technique)

        test_tagseqs = [tagseq for _, tagseq in test_ds]
        test_preds = [tagger.predict(wordseq) for wordseq, _ in test_ds]

        accuracy, precision, recall, f1, cm, class_report = evaluate_predictions(
            test_tagseqs, test_preds
        )

        print(f"Fold {fold_idx + 1} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(class_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-path", type=str, required=True)
    parser.add_argument("-k", "--k", type=int, default=5)
    parser.add_argument(
        "-t",
        "--smoothing-technique",
        choices=["laplace", "kneser_nay"],
        default="kneser_nay",
    )
    args = parser.parse_args()
    main(args)
