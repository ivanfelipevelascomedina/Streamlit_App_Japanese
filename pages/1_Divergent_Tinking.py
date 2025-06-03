# GUI DIVERGENT THINKING CODE

# IMPORT LIBRARIES
import streamlit as st
import json
import pandas as pd
from XAI_APP_utils import generate_design_output

st.set_page_config(layout="wide")  # Set wide layout for the entire app

# API Keys Setup
if st.session_state.openai_key:
    try:
        from openai import OpenAI
        st.session_state.client = OpenAI(api_key=st.session_state.openai_key)  # Store client in session_state to be able to access it from other pages
    except Exception as e:
        st.error(f"APIの初期化に失敗しました: {e}")
else:
    st.error("OpenAIのAPIキーを入力してください。")

# DEFINE THE PAGE INITIAL INFORMATION

st.title("発散的思考")

st.markdown("""
<div style="color: #1E90FF; font-size:16px;">
    <h5>重要：ワークフローに関する免責事項</h5>
    <b>発散的思考</b>ページでは、デザイン課題に対する解決策を生成します。したがって、先に<b>Main</b>ページで課題を定義してから進んでください。クエリを入力して Enter キーを押すことで定義できます。<br>
    解決策が生成されたら、<b>発散的思考フィルタリング</b>ページに進んで、それらを視覚化、分析、および最も関連性の高い選択肢を選ぶことができます。<br>
    このページで解決策が生成されていない場合、<b>フィルタリング</b>ページは機能しません。進む前に、解決策が存在することを確認してください。<br>
</div>
""", unsafe_allow_html=True)

st.markdown("""
## 使用方法：

1. **要件を選択する**:
   - 初期の課題に加えて、デザイン課題で考慮したい要件を選択してください。
   - 必要がなければ、選択しなくても構いません。

2. **解決策を生成する**:
   - 「新しいFBSオントロジーデータを生成」ボタンをクリックして、デザイン課題に対するLLMの解決策を生成します。
   - 解決策が生成された後、このページでは再表示されませんが、**フィルタリング**ページからいつでもアクセスできます。

3. **解決策を再生成する**:
   - 生成された解決策に納得できない場合は、ページ下部の「リセット」ボタンをクリックして新しい解決策を生成できます。
   - **注意**：このボタンをクリックすると、以前に生成された情報は失われます。
   - 解決策を再生成する前に、**収束的思考**ページで手動でデータを追加することも可能であることを覚えておいてください。
""")


# MAIN CODE OF THE DIVERGENT THINKING PART, HERE NUMEROUS FBS SOLUTIONS ARE GENERATED FOR THE DEFINED DESIGN PROBLEM

# Check if the OpenAI client and design problem exist in session state
if "client" in st.session_state and "design_problem" in st.session_state:
    design_problem = st.session_state.design_problem
    role_description = st.session_state.role_description_divergent
    
    # Make the user select the requirements of the design problem to be used

    # Define available requirements
    available_requirements = [
        "購入者のニーズに対応する","革新","技術的実現可能性を確保する", 
        "コストを最小限に抑える","拡張性","安全規制の順守",
        "エネルギー規制の順守","持続可能性","時間効率"
    ]

    # Normal multiselect widget
    selected_requirements = st.multiselect(
        "あなたのデザイン課題に対する要件を選択してください:",
        options=available_requirements,
        default=available_requirements,  # Start with all options selected to save time for those users who want them all
        help="「各要件をクリックすることで、その要件がChatGPTへのプロンプト情報の一部として選択されます。選択した要件は後から変更できないため、本当に必要なものだけを選択してください。変更するには、データ生成プロセスをリセットする必要があります。",
    )

    # Store the selected requirements in session state
    st.session_state.selected_requirements = selected_requirements

    # Display the selected requirements (optional, for user feedback)
    if st.session_state.selected_requirements:
        st.write(f"選択された要件: {', '.join(st.session_state.selected_requirements)}")
    else:
        st.warning("要件が選択されていません。少なくとも1つ選択してください。")
    
    # Button to generate FBS data
    if st.button("新しいFBSオントロジーデータを生成"):
        with st.spinner("FBSオントロジーデータを生成中..."):
            try:
                # Generate FBS elements for the single design problem
                fbs_entry = {
                    "Design Goal": design_problem,
                    "Requirements": selected_requirements,
                    "Functions_1": generate_design_output(design_problem, "機能", role_description, "「機能」は設計の**目的**を定義し、**何のためにあるのか**を説明します。", "エンジンの出力を上げる、燃費を改善する、排出ガスを削減する..."),
                    "Behaviors_1": generate_design_output(design_problem, "振る舞い", role_description, "「振る舞い」は設計対象の構造から導き出される**属性**を表し、**何をするか**を説明します。", "排気ガスを利用して高速回転する、空気を圧縮して質量流量を増加させる、摩擦と圧力によって熱を発生させる..."),
                    "Structures_1": generate_design_output(design_problem, "構造", role_description, "「構造」は設計を構成する**物理的な部品、材料、または構造配置**を定義し、**何で構成されているか**を説明します。これは具体的な要素であり、振る舞いの説明ではないことに注意してください。", "コンプレッサー、タービン、回転軸、鋼製ハウジング、ボールベアリング、インタークーラーパイプ..."),
                }
                
                # Save the entry to a JSON file
                file_name = "fbs_ontology_data.json"
                with open(file_name, "w") as file:
                    json.dump(fbs_entry, file, indent=4)

                # Display the generated FBS ontology data in a structured format
                st.success("FBSオントロジーデータの生成が完了しました！")

                # Display the design problem and selected requirements
                st.markdown(f"### デザイン課題:\n{fbs_entry['Design Goal']}")
                st.markdown("### 選択された要件:")
                st.write(", ".join(fbs_entry["Requirements"]))

                # Create the FBS table and a brief explanation of its elements
                st.markdown("""
                            ### FBSオントロジー要素
                            以下の表は、あなたのデザイン課題に対して提案された「機能」「振る舞い」「構造」を示しています。各列はFBSオントロジーの特定の側面を表しています：  
                            - **機能（Functions）**：そのデザインが「何のためにあるか」（目的論的側面）  
                            - **振る舞い（Behaviors）**：そのデザインが「何をするか」  
                            - **構造（Structures)**：そのデザインが「何で構成されているか」（要素、構造配置、
                            """)

                functions = fbs_entry["Functions_1"]
                behaviors = fbs_entry["Behaviors_1"]
                structures = fbs_entry["Structures_1"]

                # Display FBS elements as a table
                data = {
                    "Functions": functions,
                    "Behaviors": behaviors,
                    "Structures": structures,
                }

                # Ensure all columns have the same number of rows
                max_len = max(len(functions), len(behaviors), len(structures))
                functions += [""] * (max_len - len(functions))
                behaviors += [""] * (max_len - len(behaviors))
                structures += [""] * (max_len - len(structures))

                # Display the table
                fbs_table = pd.DataFrame(data)
                st.dataframe(fbs_table, height=400, use_container_width=True)

                # Store the table in session state to use in other pages
                st.session_state.fbs_table = fbs_table

                # Provide a download link for the JSON file
                with open(file_name, "r") as file:
                    json_data = file.read()
                st.download_button(
                    label="FBSオントロジーデータをJSONとしてダウンロード",
                    data=json_data,
                    file_name=file_name,
                    mime="application/json",
                )

            except Exception as e:
                st.error(f"FBSオントロジーデータの生成中にエラーが発生しました: {e}")

    # Add a reset button outside the try-except block
    if st.button("Reset", help="ここをクリックすると、発散的思考プロセスをリセットして最初からやり直すことができます。ただし、ここをクリックすると、データをダウンロードしない限り、すべてのデータが失われることにご注意ください。"):
        for key in st.session_state.keys():
            del st.session_state[key]

else:
    st.error("デザイン課題を入力し、OpenAIクライアントがメインページで初期化されていることを確認してください。")