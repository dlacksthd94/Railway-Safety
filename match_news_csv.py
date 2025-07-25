import pandas as pd
import numpy as np
import utils_scrape
import os
from pprint import pprint
import streamlit as st
import pandas as pd
import os

st.set_page_config(layout='wide')

DIR_DATA_ROOT = 'data'

path_form57_csv = 'data/250424 Highway-Rail Grade Crossing Incident Data (Form 57).csv'
path_df_form57_retireval = 'data/json/pdf_OpenAI_o4-mini_8/result/Huggingface_microsoft/Phi-4-mini-instruct_1_group/df_form57_retrieval.csv'
path_df_match = os.path.join(DIR_DATA_ROOT, 'df_match.csv')

def as_float(val):
    if isinstance(val, str) and val.isdigit():
        return float(val)
    else:
        return val

dict_idx_colname = {
        '1': 'Railroad Name',
        # '1a': 'Railroad Code',
        # '1b': 'Incident Number',
        '2': 'Other Railroad Name',
        # '2a': 'Other Railroad Code',
        # '2b': 'Other Incident Number',
        '3': 'Maintenance Railroad Name',
        # '3a': 'Maintenance Railroad Code',
        # '3b': 'Maintenance Incident Number',
        # '4': 'Grade Crossing ID',
        '5': ['Date', 'Month', 'Day', 'Incident Year'],
        '6': ['Time', 'Hour', 'Minute', 'AM/PM'],
        
        '7': 'Nearest Station',
        # '8': 'Subdivision',
        '9': 'County Name',
        '10': 'State Name',
        '11': 'City Name',
        '12': ['Highway Name', 'Public/Private'],
        
        '13': 'Highway User Code',
        '14': 'Estimated Vehicle Speed',
        '15': 'Vehicle Direction Code',
        '16': 'Highway User Position Code',
        
        '17': 'Equipment Involved Code',
        '18': 'Railroad Car Unit Position',
        '19': 'Equipment Struck Code',
        
        '20a': 'Hazmat Involvement Code',
        '20b': 'Hazmat Released by Code',
        '20c': ['Hazmat Released Name', 'Hazmat Released Quantity', 'Hazmat Released Measure'],

        '21': 'Temperature',
        '22': 'Visibility Code',
        '23': 'Weather Condition Code',

        '24': 'Equipment Type Code',
        '25': 'Track Type Code',
        '26': 'Track Name',
        '27': 'Track Class',
        '28': 'Number of Locomotive Units',
        '29': 'Number of Cars',

        '30': ['Train Speed', 'Estimated/Recorded Speed'],
        '31': 'Train Direction Code',
                
        # '32': [f'Crossing Warning Expanded Code {str(i)}' for i in range(1, 13)],
        # '33': 'Signaled Crossing Warning Code',
        '34': 'Roadway Condition Code',
        '35': 'Crossing Warning Location Code',
        # '36': 'Warning Connected To Signal', # no code column
        # '37': 'Crossing Illuminated', # no code column

        '38': 'User Age',
        '39': 'User Sex',
        '40': 'User Struck By Second Train', # no code column
        '41': 'Highway User Action Code',
        '42': 'Driver Passed Vehicle', # no code column
        '43': 'View Obstruction Code',
        
        '44': 'Driver Condition Code',
        '45': 'Driver In Vehicle', # no code column
        '46': ['Crossing Users Killed', 'Crossing Users Injured'],
        '47': 'Vehicle Dammge Cost',
        '48': 'Number Vehicle Occupants',
        '49': ['Employees Killed', 'Employees Injured'],
        '50': 'Number People On Train',
        # '51': 'Form 54 Filed,
        '52': ['Passengers Killed', 'Passengers Injured'],
        
        # '53a': 'Speical Study 1', #Video Taken; Video Used
        # '53b': 'Speical Study 2',
        '54': 'Narrative',
}
list_col_idx = ['1', '9', '11', '13', '38', '39']
list_col_name = [dict_idx_colname[col] for col in list_col_idx]
list_col_name_additional = ['Time', 'Highway Name', 'Nearest Station', 'Crossing Users Killed', 'Crossing Users Injured', 'Passengers Killed', 'Passengers Injured', 'Equipment Type', 'Narrative']

list_pastel = ["#FFB3BA", "#BAE1FF"]

############ hand annotation
def load_df_csv():
    df_form57_csv = pd.read_csv(path_form57_csv)
    df_form57_csv = df_form57_csv[df_form57_csv['State Name'] == 'CALIFORNIA']
    df_form57_csv['hash_id'] = df_form57_csv.apply(utils_scrape.hash_row, axis=1)
    df_form57_csv = df_form57_csv.set_index('hash_id')
    df_form57_csv['Date'] = pd.to_datetime(df_form57_csv['Date'])

    st.session_state.df_form57_csv = df_form57_csv

def load_df_match():
    if os.path.exists(path_df_match):
        df_match = pd.read_csv(path_df_match)
        df_match['pub_date'] = pd.to_datetime(df_match['pub_date'])

    else:
        df_retrieval = pd.read_csv(path_df_form57_retireval)
        df_retrieval['pub_date'] = pd.to_datetime(df_retrieval['pub_date'])

        (~df_retrieval.isin(['Unknown', "'Unknown", '(Unknown)'])).sum().loc['1':].sort_values(ascending=False)[:30] # num of non-unknown values per column

        df_retrieval['9'] = df_retrieval['9'].str.upper().str.replace('COUNTY', '').str.strip()
        df_retrieval['11'] = df_retrieval['11'].str.upper().str.replace('', '').str.strip()
        df_retrieval['13'] = df_retrieval['13'].str.replace("'", '').replace('Auto', 'A')
        df_retrieval['38'] = df_retrieval['38'].str.replace(r'\D+', '', regex=True)
        df_retrieval['39'] = df_retrieval['39'].str.replace("'", '').map({'1': 'Male', '2': 'Female', 'Male': 'Male', 'Female': 'Female', '1 (Male)': 'Male', '2 (Female)': 'Female'})
        df_retrieval['44'] = df_retrieval['44'].str.replace("'", '').map({'1': '1', '2': '2', '3': '3', 'Killed': '1', 'Injured': '2', 'Uninjured': '3', '1 (Killed)': '1', '2: Injured': '2'})

        df_match = df_retrieval.copy(deep=True)
        df_match['match'] = np.nan

        for idx, sr_retireval in df_retrieval.iterrows():
            sr_retireval = sr_retireval[['incident_id', 'pub_date'] + list_col_idx]
            incident_id = sr_retireval.pop('incident_id')
            sr_retireval = sr_retireval.apply(as_float)
            sr_form57_csv = df_form57_csv.loc[incident_id]
            sr_form57_csv = sr_form57_csv[['Date'] + list_col_name]
            mask = sr_form57_csv.values == sr_retireval.values
            if not any(mask):
                df_match.loc[idx, 'match'] = 0

        df_match.to_csv(path_df_match, index=False)

    st.session_state.df_match = df_match
    st.session_state.idx = 0

def advance_and_mark(answer: str):
    """Mark current row as Yes/No, advance index by 1."""
    i = st.session_state.idx
    st.session_state.df_match.at[i, "match"] = answer
    st.session_state.idx += 1

if "df_csv" not in st.session_state:
    load_df_csv()

if "df_match" not in st.session_state:
    load_df_match()

if 'i_color' not in st.session_state:
    st.session_state.i_color = 0
    st.session_state.color = list_pastel[st.session_state.i_color]

df_match = st.session_state.df_match
df_form57_csv = st.session_state.df_form57_csv
idx = st.session_state.idx

# Skip already-reviewed rows
while idx < len(df_match) and pd.notna(df_match.at[idx, "match"]):
    idx += 1
    st.session_state.idx = idx

if idx < len(df_match):
    col_left, col_right = st.columns(2)

    with col_left:
        sr_match = df_match.loc[idx, ['incident_id', 'pub_date'] + list_col_idx]
        incident_id = sr_match.pop('incident_id')
        if idx > 0:
            incident_id_prev = df_match.loc[idx - 1, 'incident_id']
        else:
            incident_id_prev = None
        sr_match = sr_match.apply(as_float)
        sr_match = pd.concat([sr_match, pd.Series([np.nan] * len(list_col_name_additional))])
        sr_form57_csv = df_form57_csv.loc[incident_id]
        sr_form57_csv = sr_form57_csv[['Date'] + list_col_name + list_col_name_additional]
        mask = sr_form57_csv.values == sr_match.values
        with st.form('### Comparison'):
            df_compare = pd.DataFrame({'csv': sr_form57_csv.values, 'news': sr_match.values, 'match': mask}, index=sr_form57_csv.index)
            day_name_csv, day_name_news = sr_form57_csv['Date'].day_name(), sr_match['pub_date'].day_name()
            df_compare.loc['day_name'] = [day_name_csv, day_name_news, day_name_csv == day_name_news]
            
            if incident_id != incident_id_prev:
                st.session_state.i_color += 1
                st.session_state.color = list_pastel[st.session_state.i_color % 2]
                df_match.to_csv(path_df_match, index=False)
            else:
                st.session_state.color = st.session_state.color
            df_compare_styled = df_compare.style.set_properties(**{"background-color": st.session_state.color}, subset=["csv"])
            st.dataframe(df_compare_styled, height=int(35.5 * (df_compare.shape[0] + 1)))
            
            _ = st.form_submit_button("Save changes", disabled=True)

        st.markdown(f"### [{idx}] Does this news match the record?")

        col_yes, col_no, col_unsure, col_including = st.columns(4)
        col_yes.button("Yes", on_click=advance_and_mark, args=(1,))
        col_no.button("No",  on_click=advance_and_mark, args=(0,))
        col_unsure.button("Unsure",  on_click=advance_and_mark, args=(2,))
        col_including.button("Including",  on_click=advance_and_mark, args=(3,))

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("💾 Save updates"):
                st.session_state.df_match.to_csv(path_df_match, index=False)
                st.success(f"Saved to **{path_df_match}**!")
    
    with col_right:
        st.markdown("### News article")
        st.write(df_match.loc[idx, 'rd_url'])

else:
    st.success("✅ All news reviewed!")
