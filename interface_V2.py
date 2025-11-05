from typing import Any, Dict
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from baselines.raw_sentiment import baseline_sentiment_json
from config import LLM_CONFIG
from llm_interface import OpenAILLM
import pandas as pd
from analysis.sentiment import metamind_sentiment_json
from analysis.recommendation_text import recommendation_text_from_result

def ensure_api_key(config: Dict[str, Any]) -> None:
    api_key = config.get("api_key")
    if not api_key or api_key in {"your_api_key_here", "replace_me"}:
        raise ValueError(
            "OpenAI API key missing. Set OPENAI_API_KEY or update config.py."
        )

@st.cache_resource
def load_llm() -> OpenAILLM:
    ensure_api_key(LLM_CONFIG)
    return OpenAILLM(LLM_CONFIG)

def main():
    st.set_page_config(page_title="MetaMind Sentiment Demo", layout="wide")
    st.title(" MetaMind Sentiment Demo")

    # load the csv of reviews
    df = pd.read_csv("data/clean/Iphone.csv", header=None, names=["Review"])

    # display the tale
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection("single", use_checkbox=False)
    grid_options = gb.build()

    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        height=250,
        theme="streamlit",
    )

    selected_rows = grid_response.get("selected_rows")
    if selected_rows is None:
        selected_rows = []

    if isinstance(selected_rows, pd.DataFrame):
        has_selection = not selected_rows.empty
    else:
        has_selection = len(selected_rows) > 0

    if has_selection:
        if isinstance(selected_rows, pd.DataFrame):
            selected_review = selected_rows.iloc[0]["Review"]
        else:
            selected_review = selected_rows[0]["Review"]

        if st.button("Analyze sentiment", type="primary"):
            llm = load_llm()
            context = []
            hypotheses = []

            with st.spinner("Running sentiment analysis..."):
                baseline_result = baseline_sentiment_json(llm, selected_review, context, max_retries=1)
                metamind_result = metamind_sentiment_json(llm, selected_review, context, hypotheses)
                recommendation_text = recommendation_text_from_result(metamind_result)

            left, right = st.columns(2)
            with left:
                st.subheader("Baseline output")
                st.json(baseline_result)
            with right:
                st.subheader("MetaMind output")
                st.json(metamind_result)
                st.subheader("Recommendation")
                st.write(recommendation_text)

            st.success("Analysis complete")

    else:
        st.info("Click on a review in the table to select it.")


if __name__ == "__main__":
    main()
