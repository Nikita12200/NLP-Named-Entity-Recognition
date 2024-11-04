from datasets import load_dataset

def main():
    try:
        # Load the CONLL2003 dataset
        tagss =[]
        tokenss =[]
        data = load_dataset("conll2003")
        data_validation = data["validation"]
        for d in data_validation:
            tags = d["ner_tags"]
            tokens = d["tokens"]
            tagss.append(tags)
            tokenss.append(tokens)
        # Print the first 10 entries of the validation dataset
        print(tagss[10:15])
        print(tokenss[10:15])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
