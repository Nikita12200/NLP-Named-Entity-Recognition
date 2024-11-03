from ner_tag import ner_tag
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from datasets import load_dataset
import argparse
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

def evaluate_predictions(Y_test, Y_pred, all_labels):

    # Flatten test_tagseqs and preds while ensuring each tag is a string
    test_tags_flat = Y_test
    preds_flat =Y_pred

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
    plt.savefig("assets/heatmap/confusion_matrix_heatmap.png")  # Save heatmap to a file
    plt.close()
    # plt.show()

    return accuracy, precision, recall, f, cm, class_report

def main(args: argparse.Namespace):
    # Load the test dataset
    # data = load_dataset("Davlan/conll2003_noMISC")
    data = load_dataset("conll2003")

    data_test = data["test"]

    # Initialize the NER tagger
    ner_tagger = ner_tag()

    # Prepare the test data
    words_test,X_test, Y_test = ner_tagger.createData(data_test)
    Y_test = Y_test.astype(int)
    # Load the model and scaler
    with open(args.nei_model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(args.scaler_model_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Make predictions on the test set
    Y_pred, _ = ner_tagger.infer(words_test, args.nei_model_path, args.scaler_model_path,flag = 1)
    print(Y_pred[:10])
    print(Y_test[:10])
    if isinstance(Y_pred[0], (list, np.ndarray)):
        Y_pred = [tag for tags in Y_pred for tag in tags]
        Y_test = [tag for tags in Y_test for tag in tags]

    # Collect all unique labels
    all_labels = [0, 1]

    accuracy, precision, recall, f, cm, class_report = evaluate_predictions(
            Y_test, Y_pred, all_labels
        )
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-Score: {f:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(class_report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nei-model-path", type=str, required=True, help="Path to the NEI model file.")
    parser.add_argument("-s", "--scaler-model-path", type=str, required=True, help="Path to the scaler model file.")
    args = parser.parse_args()
    main(args)
