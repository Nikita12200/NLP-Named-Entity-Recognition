import streamlit as st
import argparse
import pycrfsuite
from src.crf_pos_tag import CRFPosTagger


# def process_string(input_string: str) -> str:
#     return input_string[::-1]


# @st.cache_resource
# def load_model(path: str) -> CRFPosTagger:
#     return CRFPosTagger.load_model(path)


def main(args: argparse.Namespace):
    st.title("CRF Based Parts of Speech Tagging")
    st.image("./assets/logo.png")

    # Load the model
    # model = load_model(args.model_path)
    model = pycrfsuite.Tagger()
    model.open(args.model_path)
    # Input text
    wordseq = st.text_input("Enter a sentence:", "")
    
    if wordseq:
        # Convert sentence into list of words (list of lists for predict_crf)
        sentences = [wordseq.split()]

        # Predict tags
        crf_tagger = CRFPosTagger()

        tagseqs = crf_tagger.predict_crf(sentences, args.model_path)
        
        if tagseqs:
            tagseq = tagseqs[0]  # Get the tags for the single input sentence
            
            output = ""
            for word, tag in zip(sentences[0], tagseq):
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
    parser.add_argument("-m", "--model-path", type=str)
    args = parser.parse_args()

    main(args)
