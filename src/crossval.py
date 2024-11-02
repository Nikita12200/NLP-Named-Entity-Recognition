from ner_tag import ner_tag
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from datasets import load_dataset
import argparse
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def main(args: argparse.Namespace):
    # Load the test dataset
    data = load_dataset("Davlan/conll2003_noMISC")
    data_test = data["test"]
    # print(data_test[:5])
    # Initialize the NER tagger
    ner_tagger = ner_tag()

    # Prepare the test data
    words_test,X_test, Y_test = ner_tagger.createData(data_test)
    # print(words_test[:5])
    # print(Y_test[:5])
    # Load the model and scaler
    with open(args.nei_model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(args.scaler_model_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Make predictions on the test set
    Y_pred, _ = ner_tagger.infer(words_test, args.nei_model_path, args.scaler_model_path,flag = 1)
    
    print("PREDICTED SET Y LENGTH:",len(Y_pred))
    # Debugging: Print unique values in Y_test and Y_pred
    print("Unique values in Y_test:", np.unique(Y_test))
    print("Unique values in Y_pred:", np.unique(Y_pred))

    # Generate and print classification report
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred, target_names=["Non-Entity", "Entity"]))

    # Display accuracy score
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Generate confusion matrix
    cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1])
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Entity", "Entity"], yticklabels=["Non-Entity", "Entity"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nei-model-path", type=str, required=True, help="Path to the NEI model file.")
    parser.add_argument("-s", "--scaler-model-path", type=str, required=True, help="Path to the scaler model file.")
    args = parser.parse_args()
    main(args)
