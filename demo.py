from modules import utils, build_config, extract_keywords
import streamlit as st
import pandas as pd


args = {
    "c_api": "OpenAI",
    "c_model": "o4-mini",
    "c_n_generate": 4,
    "c_json_source": "img",
    "r_api": "Huggingface",
    "r_model": "microsoft/phi-4",
    "r_n_generate": 1,
    "r_question_batch": "group"
}
cfg = build_config(args)

st.set_page_config(layout="wide")

# -------------------------------
# Load data
# -------------------------------

df_retrieval = utils.prepare_df_retrieval(cfg)
dict_col_indexing = utils.prepare_dict_col_indexing(cfg)
dict_idx_mapping, dict_idx_mapping_inverse = utils.prepare_dict_idx_mapping(cfg)
dict_form57 = utils.prepare_dict_form57(cfg)
dict_form57_group = utils.prepare_dict_form57_group(cfg)

dict_group_field_cols = {}
for name, group in dict_form57_group.items():
    dict_field_cols_temp = {}
    content_idx = df_retrieval.columns.get_loc('content')
    for field in group:
        cols_temp = []
        for col in df_retrieval.columns[content_idx + 1:]: # type: ignore
            if field == col.split('_')[0]:
                cols_temp.append(col)
        dict_field_cols_temp[field] = cols_temp
    dict_group_field_cols[name] = dict_field_cols_temp

st.session_state["news_df"] = df_retrieval

df = st.session_state.get("news_df", None)
if df is None:
    st.error("Please load your DataFrame into st.session_state['news_df']")
    st.stop()

# -------------------------------
# Sidebar — Date Range Filter
# -------------------------------
st.sidebar.header("Filters")

min_date = df['pub_date'].min().date()
max_date = df['pub_date'].max().date() + pd.Timedelta(days=1)

date_range = st.sidebar.date_input(
    "News Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

start_date, end_date = date_range if isinstance(date_range, tuple) else (min_date, max_date) # type: ignore
filtered_df = df[(df['pub_date'] >= pd.Timestamp(start_date)) & (df['pub_date'] <= pd.Timestamp(end_date))]

# -------------------------------
# PAGE LAYOUT
# -------------------------------
col_left, col_right = st.columns([1, 1])   # Left wider than right

with col_left:
    upper_left = st.container()
    lower_left = st.container()

with col_right:
    right_panel = st.container()

# Remember selection
if "selected_news_id" not in st.session_state:
    st.session_state.selected_news_id = None


# -------------------------------
# UPPER LEFT — News List
# -------------------------------
with upper_left:
    options = {
        f"{row.pub_date.date()} — {row.title[:80]}...": row.news_id
        for _, row in filtered_df.iterrows()
    }

    if options:
        selected_label = st.selectbox(
            "Select a news article:",
            list(options.keys()),
            index=0
        )
        st.session_state.selected_news_id = options[selected_label]
    else:
        st.info("No news found in selected date range.")


# -------------------------------
# Get selected news item
# -------------------------------
selected = None
if st.session_state.selected_news_id is not None:
    selected = df[df["news_id"] == st.session_state.selected_news_id].iloc[0]


# -------------------------------
# LOWER LEFT — Embedded Original Page
# -------------------------------
with lower_left:
    st.subheader("Original Article Preview")
    
    if selected is not None:
        st.components.v1.iframe(src=selected["url"], scrolling=True, height=800)

        # st.markdown(f"### {selected['title']}")
        # st.write(f"**Published:** {selected['pub_date']}")
        # st.markdown("---")
        # st.write(selected["content"])
    else:
        st.info("Select a news item to preview the webpage.")


# -------------------------------
# RIGHT PANEL — News Details
# -------------------------------
with right_panel:
    st.subheader("News Report Details")

    if selected is not None:
        content_idx = selected.index.get_loc('content')
        for group, field_cols in dict_group_field_cols.items():
            st.markdown(f"#### {group}")
            for field, cols in field_cols.items():
                field_name = dict_form57[field]["name"]
                if len(cols) == 1:
                    retrieved_value = selected[cols[0]]
                    st.markdown(f'**{field}. {field_name}**: {retrieved_value}')
                else:
                    st.markdown(f'**{field}. {field_name}**')
                    for i, col in enumerate(cols):
                        retrieved_value = selected[col]
                        subfield_name = list(dict_form57[field]['answer_places'].keys())[i]
                        st.markdown(f"- {subfield_name}: {retrieved_value}")
                
                
        #         if field in dict_idx_mapping_inverse:
        #             field_name = dict_idx_mapping_inverse[field]
        #             field_value = selected[field] if pd.notna(selected[field]) else "N/A"
        #             st.markdown(f"**{field_name}**: {field_value}")
        # for field in selected.index[content_idx + 1:]:
        #     if field in dict_idx_mapping_inverse:
        #         dict_idx_mapping_inverse[field]
        #     if field in dict_col_indexing:
        #         dict_col_indexing[field]
        # st.write(selected['1'])
    else:
        st.info("Select a news item to view details.")