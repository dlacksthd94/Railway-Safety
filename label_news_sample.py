import pandas as pd
import streamlit as st
import os

DATA_FOLDER = 'data/'
FN_DF_FILTER = 'df_news_filter.csv'
FN_DF_SAMPLE = 'df_news_label_sample.csv'
COLUMNS_CONTENT = ['np_url', 'tf_url', 'rd_url', 'gs_url', 'np_html', 'tf_html', 'rd_html', 'gs_html']

path_df_filter = DATA_FOLDER + FN_DF_FILTER
path_df_sample = DATA_FOLDER + FN_DF_SAMPLE

def load_data() -> pd.DataFrame:
    if os.path.exists(path_df_sample):
        df = pd.read_csv(path_df_sample, index_col=[0,1])
        # df['label'].value_counts()
    else:
        df_filter = pd.read_csv(path_df_filter)
        n_sample = 200
        df = df_filter[COLUMNS_CONTENT].stack().to_frame('content').sample(n_sample, replace=False)
        df['label'] = float('nan')
        df.to_csv(path_df_sample)
    return df

if 'df' not in st.session_state:
    st.session_state.df = load_data()

def next_unlabeled(start_pos: int) -> int:
    df = st.session_state.df
    pos = start_pos
    while pos < len(df) and pd.notna(df.iloc[pos]['label']):
        pos += 1
    return pos

if 'idx' not in st.session_state:
    st.session_state.idx = next_unlabeled(0)

def mark_and_advance(choice: str):
    df = st.session_state.df
    idx_tuple = df.index[st.session_state.idx]
    df.at[idx_tuple, 'label'] = choice
    st.session_state.idx = next_unlabeled(st.session_state.idx + 1)

st.title("🚂 Train-Accident Labeling Tool")

if st.session_state.idx < len(st.session_state.df):
    idx = st.session_state.idx
    st.markdown(f"**Article {idx+1}/{len(st.session_state.df)}:**")
    st.write(st.session_state.df.iloc[idx]['content'])

    st.markdown("**Is this article reporting a train accident?**")
    col1, col2 = st.columns(2)
    with col1:
        st.button("✅ YES", on_click=mark_and_advance, args=("YES",))
    with col2:
        st.button("❌ NO",  on_click=mark_and_advance, args=("NO",))
else:
    st.success("🎉 All articles labeled!")

st.markdown("---")
if st.button("💾 Save labels"):
    st.session_state.df.to_csv(path_df_sample)
    st.success(f"Saved to {path_df_sample}")
    csv_bytes = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv_bytes,
                       file_name=FN_DF_SAMPLE, mime="text/csv")