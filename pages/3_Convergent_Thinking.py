# GUI CONVERGENT THINKING CODE

# IMPORT LIBRARIES
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")  # Set wide layout for the entire app

# DEFINE THE PAGE INITIAL INFORMATION

st.title("収束的思考")

st.markdown("""
<div style="color: #1E90FF; font-size:16px;">
    <h5>重要：ワークフローに関する免責事項</h5>
    <b>収束的思考</b>ページでは、デザイン課題に対して生成された解決策を精査・フィルタリングすることができます。<b>発散的思考フィルタリング</b>の確認と各設計案の分析を行った後、興味のある解決策を選択したり、新たに追加したりすることができます。<br>
    続行する前に、<b>発散的思考</b>ページで<b>機能</b>、<b>振る舞い</b>、<b>構造</b>の各リストが正しく初期化されていることを確認してください。これらのリストが初期化されていない場合、このページは正しく動作しません。<br>
</div>
""", unsafe_allow_html=True)

st.markdown("""
## 使用方法：

1. **オプションを選択する**:  
   - 最終的な設計に含めたいオプションを選ぶには、各オプションの右側にある「選択」ボックスをクリックしてください。

2. **新しいオプションを追加する**:  
   - 各テーブルに新しい行を追加するには、テーブル下部の「行を追加」ボタンをクリックしてください。  
   - 誤って不要な行を追加しても心配はいりません。その行の右側にある「選択」ボックスが未チェックであれば、最終結果に影響はありません。  
   - 既存の行または追加した行を編集するには、その行をクリックし、編集したい内容を入力してください。

3. **変更を反映させる**:  
   - 変更を反映させるには、ページ下部の「データを更新」ボタンをクリックしてください。クリックしない場合、行った変更は反映されません。  
   - 正しく反映されているか確認したい場合は、「最終結果」タブで表示されているオプションが自分の選択と一致しているかを確認してください。
""")

# Step 1: Check if data exists in session state for functions, behaviors and structures
if "convergent_thinking_functions_data" in st.session_state:
    functions_data = st.session_state["convergent_thinking_functions_data"]
else:
    # Initialize from `functions_list` if available
    if "functions_list" in st.session_state:
        functions_data = pd.DataFrame({
            "Option": st.session_state["functions_list"],
            "Selection": [False] * len(st.session_state["functions_list"])  # Default widget values to False
        })
        # Save the initialized data to session state
        st.session_state["convergent_thinking_functions_data"] = functions_data
    else:
        st.error(f"セッション状態に機能リストが存在しません。続行する前に初期化してください。")

if "convergent_thinking_behaviors_data" in st.session_state:
    behaviors_data = st.session_state["convergent_thinking_behaviors_data"]
else:
    # Initialize from `functions_list` if available
    if "behaviors_list" in st.session_state:
        behaviors_data = pd.DataFrame({
            "Option": st.session_state["behaviors_list"],
            "Selection": [False] * len(st.session_state["behaviors_list"])  # Default widget values to False
        })
        # Save the initialized data to session state
        st.session_state["convergent_thinking_behaviors_data"] = behaviors_data
    else:
        st.error(f"セッション状態に振る舞いリストが存在しません。続行する前に初期化してください。")

if "convergent_thinking_structures_data" in st.session_state:
    structures_data = st.session_state["convergent_thinking_structures_data"]
else:
    # Initialize from `functions_list` if available
    if "structures_list" in st.session_state:
        structures_data = pd.DataFrame({
            "Option": st.session_state["structures_list"],
            "Selection": [False] * len(st.session_state["structures_list"])  # Default widget values to False
        })
        # Save the initialized data to session state
        st.session_state["convergent_thinking_structures_data"] = structures_data
    else:
        st.error(f"セッション状態に構造リストが存在しません。続行する前に初期化してください。")

# Step 2: Display the DataFrame using st.data_editor for the functions, behaviors and structures and add a row with default `selection=False` in any of the chosen tables
# Step 2.1: Functions
st.write("機能テーブル")
edited_functions_data = st.data_editor(functions_data, num_rows="dynamic", use_container_width=True, key="functions_table_editor")
if st.button("機能に行を追加"):
    # Add a new row to the DataFrame
    new_functions_row = pd.DataFrame([{"Option": "", "Selection": False}])
    functions_data = pd.concat([edited_functions_data, new_functions_row])

    # Store the updated DataFrame in session state
    st.session_state["convergent_thinking_functions_data"] = functions_data
    st.write("新しい機能の行が追加されました")

# Step 2.2: Behaviors
st.write("振る舞いテーブル")
edited_behaviors_data = st.data_editor(behaviors_data, num_rows="dynamic", use_container_width=True, key="behaviors_table_editor")
if st.button("振る舞いに行を追加"):
    # Add a new row to the DataFrame
    new_behaviors_row = pd.DataFrame([{"Option": "", "Selection": False}])
    behaviors_data = pd.concat([edited_behaviors_data, new_behaviors_row])

    # Store the updated DataFrame in session state
    st.session_state["convergent_thinking_behaviors_data"] = behaviors_data
    st.write("新しい振る舞いの行が追加されました")

# Step 2.3 Structures
st.write("構造テーブル")
edited_structures_data = st.data_editor(structures_data, num_rows="dynamic", use_container_width=True, key="structures_table_editor")
if st.button("構造に行を追加"):
    # Add a new row to the DataFrame
    new_structures_row = pd.DataFrame([{"Option": "", "Selection": False}])
    structures_data = pd.concat([edited_structures_data, new_structures_row])

    # Store the updated DataFrame in session state
    st.session_state["convergent_thinking_structures_data"] = structures_data
    st.write("新しい構造の行が追加されました")

# Step 3: Button to force display updated data
if st.button("データを更新"):
    # Store the updated DataFrame in session state for functions behaviors and structures
    st.session_state["convergent_thinking_functions_data"] = edited_functions_data
    st.session_state["convergent_thinking_behaviors_data"] = edited_behaviors_data
    st.session_state["convergent_thinking_structures_data"] = edited_structures_data
    st.write("データが更新されました")