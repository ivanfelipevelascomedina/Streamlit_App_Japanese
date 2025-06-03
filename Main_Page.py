# GUI MAIN CODE

# IMPORT LIBRARIES
import streamlit as st

# DEFINE THE PAGES THAT THE GUI WILL HAVE
st.set_page_config(
    page_title="Homepage", # Set this page as the main page of the app
    layout="wide" # Set wide layout for the entire app
)

#DEFINE THE SIDEBAR INFORMATION

st.sidebar.success("デザインプロセスのどの段階に取り組みたいかを選択してください。")

# API Keys Setup
# 1. Check if the key is already in session state and initialize it if it's not
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""  # Use a session_state to be able to access it from the other pages
# 2. API Key Input
openai_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.openai_key)
# 3. Update session state with the entered key
if openai_key:
    st.session_state.openai_key = openai_key
# 4. Initialize APIs if the key is provided in session state
if st.session_state.openai_key:
    try:
        from openai import OpenAI
        st.session_state.client = OpenAI(api_key=st.session_state.openai_key)  # Store client in session_state to be able to access it from other pages
        st.success("APIの初期化に成功しました！")
    except Exception as e:
        st.error(f"APIの初期化に失敗しました: {e}")
else:
    st.error("OpenAIのAPIキーを入力してください。")

# DEFINE THE PAGE INITIAL INFORMATION

st.title("Main Page")

st.markdown("""
<div style="color: #1E90FF; font-size:16px;">
    <h5>重要、デザインプロセスにおけるLLMの使用に関する免責事項</h5>
    このデザインプロセスでは、大規模言語モデルを使用して提案、計算、その他の出力を生成しています。これらのモデルは有用な支援を提供しますが、その結果は確率的なアルゴリズムに基づいて生成されており、常に正確であるとは限らず、最適または完全であるとも限りません。利用者は以下の点に留意してください：
    <ol>
        <li><b>すべての出力を確認する：</b> LLMによって提供されたすべての提案や計算結果を慎重に評価してください。</li>
        <li><b>専門的な判断を行う：</b> 最終的な判断は、個人の専門知識およびドメイン知識に基づいて行ってください。</li>
        <li><b>重要なデータを検証する：</b> 重要な計算やデータは実装前に独自に検証してください。</li>
        <li><b>限界を理解する：</b> LLMはすべての変数、制約、またはタスク固有の微妙な点を考慮しているわけではないことを理解してください。</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.markdown("""
## 使用方法：

1. **デザイン課題を定義する**:
   - デザインプロセスを始めるには、まずデザイン課題を入力して Enter キーを押してください。
   - テキストパネルの右側にある「Enter キーを押して適用」という小さなテキストが消えていることを確認してください。表示されたままだと変更は適用されません。
   - 間違ってデザイン課題を正しく書けなかった場合は、ページを再読み込みして書き直してください。この値は後続のページに影響するため、後から変更することはできません。

2. **デザインプロセスを続ける**:
   - デザイン課題を定義したら、次のページへ進んでください。
   - このアプリの理論的な背景や使用方法についてさらに知りたい場合は、**情報**ページをクリックすると詳しい説明が表示されます。
            
""")

if "client" in st.session_state:
    # Store role descriptions in session state
    if "role_description_divergent" not in st.session_state:
        st.session_state.role_description_divergent = "あなたは、FBSオントロジーの設計課題に対して、数多くの革新的な設計提案を行うことができる経験豊富なデザイナーです。"
    if "role_description_convergent" not in st.session_state:
        st.session_state.role_description_convergent = "あなたは、設計提案を「悪い」「良くない」「普通」「良い」「優れている」と分類できるデザインの専門家です。"
    # Get the design problem description and store it in session state only once, to avoid its value being modified if the user comes back to this page
    if "design_problem" not in st.session_state:
        # Ask for input only if not set
        input_value = st.text_input("あなたのデザイン課題を入力してください：")
        if input_value:  # Only set session state if the user provides input
            st.session_state.design_problem = input_value
    else:
        # Display the stored value without modification
        st.write(f"**デザイン課題**: {st.session_state.design_problem}")
