import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
from orchestrator import run_research


st.title("Quant Research OS v3")

tickers = st.text_input("Tickers", "AAPL TSLA SPY").split()

if st.button("Run"):

    output = run_research(tickers)

    st.subheader("Leaderboard")
    st.dataframe(output["leaderboard"])

    st.subheader("DB History")
    st.dataframe(pd.DataFrame(output["db_leaderboard"]))
