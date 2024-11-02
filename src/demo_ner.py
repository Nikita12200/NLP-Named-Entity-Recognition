import streamlit as st
import argparse
from ner_tag import ner_tag

def main(args: argparse.Namespace):
    st.title("Named Entity Recognition using SVM Binary Classification")
    st.image("./assets/logo.png")

    # Input text
    wordseq = st.text_input("Enter a sentence:", "")
    
    if wordseq:
        # Initialize the NER tagger
        ner_tagger = ner_tag()
        # tokens = wordseq.split()  
        # print(f"Tokenized input: {tokens}") 
        # Call infer with model paths and sentence input
        pred, tokens= ner_tagger.infer(wordseq, args.nei_model_path, args.scaler_model_path,flag=0)
        
        # Adjust the check for `pred`
        if len(pred) > 0:  # Check if `pred` has any elements
            output = ""
            for word, tag in zip(tokens, pred):
                output += f'<span style="background-color:#891652; border-radius: 5px;">{word}</span>_<span style="background-color:#7755ff; border-radius: 5px;">{tag}</span> '
            output = output.strip()

            st.write("<h2>OUTPUT:</h2>", unsafe_allow_html=True)
            st.write(
                '<h4 style="border: 2px solid #FB6D48; border-radius: 25px; text-align: center; line-height: 1.6">'
                + output
                + "</h4>",
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nei-model-path", type=str, required=True, help="Path to the NEI model file.")
    parser.add_argument("-s", "--scaler-model-path", type=str, required=True, help="Path to the scaler model file.")
    args = parser.parse_args()

    main(args)
