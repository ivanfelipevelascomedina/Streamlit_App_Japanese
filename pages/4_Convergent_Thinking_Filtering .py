# GUI FILTERING CODE

# IMPORT LIBRARIES
import streamlit as st
from XAI_APP_utils import answer_generation, extract_probs_information, substitute_tokens, calculate_prob_difference, visualize_scores, clean_tokens, calculate_stopping_condition, calculate_feature_importance
from transformers import  RobertaForMaskedLM, RobertaTokenizer, BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, AutoModel
from openai import OpenAI

#@st.cache_resource
#def load_model():
#    tokenizer = AutoTokenizer.from_pretrained("ku-nlp/deberta-v2-tiny-japanese")
#    model = AutoModelForMaskedLM.from_pretrained("ku-nlp/deberta-v2-tiny-japanese")
#    #model.eval()  # Set the model to evaluation mode
#    return tokenizer, model

st.set_page_config(layout="wide")  # Set wide layout for the entire app

# API Keys Setup
if st.session_state.openai_key:
    try:
        from openai import OpenAI
        st.session_state.client = OpenAI(api_key=st.session_state.openai_key)  # Store client in session_state to be able to access it from other pages
    except Exception as e:
        st.error(f"APIの初期化に失敗しました: {e}")
else:
    st.error("OpenAIのAPIキーを入力してください")

# DEFINE THE PAGE INITIAL INFORMATION

st.title("最終結果")

st.markdown("""
<div style="color: #1E90FF; font-size:16px;">
    <h5>重要：ワークフローに関する免責事項</h5>
    <b>収束的思考フィルタリング</b>ページでは、LLMが各提案をどのように評価しているか、またその評価の根拠となる要素を確認することができます。<br>
    最終的な設計に含めたい<b>機能</b>、<b>振る舞い</b>、<b>構造</b>を選択した後は、ここでそれらをより深く理解し、必要に応じて修正を行うことが可能です。<br>
    続行する前に、<b>収束的思考</b>ページで<b>機能</b>、<b>振る舞い</b>、<b>構造</b>のリストが正しく選択されていることを確認してください。これらのリストが選択されていない場合、このページは正しく動作しません。<br>
</div>
""", unsafe_allow_html=True)

st.markdown("""
## 使用方法：

1. **選択したオプションに対するLLMの意見を表示する**:  
   - 選択したオプションに対するLLMの解釈を確認するには、該当するテーブルの下にあるドロップダウンリストからそのオプションを1つ選んでください。  
   - 一度に選択できるのは1つのみです（機能、振る舞い、または構造）。複数選択した場合、結果は表示されません。  
   - 結果を表示するには、ページ下部の「選択項目の特徴重要度を表示」ボタンをクリックしてください。初回は計算に数分かかることがありますが、一度計算されれば、以後はすぐに結果が表示されます。

2. **モデルの出力を理解する**:  
   - 各選択肢について、モデルがどのようにその提案を評価しているか、どの情報を重視しているかを視覚的に確認できます。  
   - **意見の分類**は以下のとおりです：  
     · **bad (1/5)：** 解決策が非効果的・非現実的・設計目標と無関係な場合  
     · **poor (2/5)：** コストが高い、実現性が低い、性能が弱いなどの重大な欠点がある場合  
     · **regular (3/5)：** 機能するが独自性や最適化、明確な利点に欠ける場合  
     · **good (4/5)：** よく設計されており、実現可能で有用だが、軽微な欠点がある場合  
     · **bad (5/5)：** 非常に革新的で、完全に実現可能かつ設計目標を大きく改善する場合  
   - モデルが**各トークン（LLMが捉える最小単位、通常は単語程度の長さ）にどれだけ重みを置いているか**を理解するために、モデルに与えた質問とともに色分けされたグラフが表示されます。色が濃いほど、そのトークンに対してモデルがより強く注目していることを意味します。
""")

# CHECK IF ROBERTA IS ALREADY LOADED IN THE SESSION STATE AND DO IT IN CASE IT IS NOT

if "roberta_tokenizer" not in st.session_state or "roberta_model" not in st.session_state:
    st.write("RoBERTaモデルとトークナイザーを読み込んでいます。数秒かかる場合があります…")
    # Load pre-trained RoBERTa model and tokenizer
    #tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    #model = RobertaForMaskedLM.from_pretrained('roberta-base')
    tokenizer = AutoTokenizer.from_pretrained("ku-nlp/deberta-v2-tiny-japanese")
    model = AutoModelForMaskedLM.from_pretrained("ku-nlp/deberta-v2-tiny-japanese")
    #tokenizer = AutoTokenizer.from_pretrained("ku-nlp/deberta-v2-tiny-japanese-char-wwm")
    #model = AutoModelForMaskedLM.from_pretrained("ku-nlp/deberta-v2-tiny-japanese-char-wwm")
    model.eval()  # Set the model to evaluation mode
    # Downloads and caches the model
    #tokenizer, model = load_model()

    # Store the model and tokenizer in session state
    st.session_state["roberta_tokenizer"] = tokenizer
    st.session_state["roberta_model"] = model
    st.write("RoBERTaモデルとトークナイザーの読み込みに成功しました！")
else:
    st.write("RoBERTaモデルとトークナイザーはすでに読み込まれています。")
    tokenizer = st.session_state["roberta_tokenizer"]
    model = st.session_state["roberta_model"]

# LOAD THE CHOSEN DATA AND LET THE USERS CHOOSE FOR WHICH ONE THEY WANT TO SEE THE REAGENT FEATURE IMPORTANCE

# FUNCTIONS
# Check if the convergent thinking data exists in session state and load it in case it does not
if "convergent_thinking_functions_data" in st.session_state:
    # Access the stored DataFrame
    convergent_thinking_functions_data = st.session_state["convergent_thinking_functions_data"]
    
    # Filter rows where `is_widget` is True
    convergent_thinking_filtered_functions_data = convergent_thinking_functions_data[convergent_thinking_functions_data["Selection"] == True]
    
    # Display the filtered rows
    st.write("選択された機能")
    st.write(convergent_thinking_filtered_functions_data)

    # Allow user to choose one of the selected rows
    if not convergent_thinking_filtered_functions_data.empty:
        convergent_thinking_filtered_functions_data_with_empty = [""] + convergent_thinking_filtered_functions_data["Option"].tolist()
        selected_function = st.selectbox(
            "機能を選択してください:",
            convergent_thinking_filtered_functions_data_with_empty
        )
        st.write(f"選択された機能: {selected_function}")
    else:
        st.write("選択された機能がありません。")

else:
    # Inform the user if the data is not available
    st.write("データが見つかりません。他のページで収束的思考テーブルを完成させてください。")

# BEHAVIORS
# Check if the convergent thinking data exists in session state and load it in case it does not
if "convergent_thinking_behaviors_data" in st.session_state:
    # Access the stored DataFrame
    convergent_thinking_behaviors_data = st.session_state["convergent_thinking_behaviors_data"]
    
    # Filter rows where `is_widget` is True
    convergent_thinking_filtered_behaviors_data = convergent_thinking_behaviors_data[convergent_thinking_behaviors_data["Selection"] == True]
    
    # Display the filtered rows
    st.write("選択された振る舞い")
    st.write(convergent_thinking_filtered_behaviors_data)

    # Allow user to choose one of the selected rows
    if not convergent_thinking_behaviors_data.empty:
        convergent_thinking_filtered_behaviors_data_with_empty = [""] + convergent_thinking_filtered_behaviors_data["Option"].tolist()
        selected_behavior = st.selectbox(
            "振る舞いを選択してください:",
            convergent_thinking_filtered_behaviors_data_with_empty
        )
        st.write(f"選択された振る舞い: {selected_behavior}")
    else:
        st.write("選択された機能がありません。")

else:
    # Inform the user if the data is not available and load it in case it does not
    st.write("データが見つかりません。他のページで収束的思考の振る舞いテーブルを完成させてください。")

# STRUCTURES
# Check if the convergent thinking data exists in session state
if "convergent_thinking_structures_data" in st.session_state:
    # Access the stored DataFrame
    convergent_thinking_structures_data = st.session_state["convergent_thinking_structures_data"]
    
    # Filter rows where `is_widget` is True
    convergent_thinking_filtered_structures_data = convergent_thinking_structures_data[convergent_thinking_structures_data["Selection"] == True]
    
    # Display the filtered rows
    st.write("選択された構造")
    st.write(convergent_thinking_filtered_structures_data)

    # Allow user to choose one of the selected rows
    if not convergent_thinking_filtered_structures_data.empty:
        convergent_thinking_filtered_structures_data_with_empty = [""] + convergent_thinking_filtered_structures_data["Option"].tolist()
        selected_structure = st.selectbox(
            "構造を選択してください:",
            convergent_thinking_filtered_structures_data_with_empty
        )
        st.write(f"選択された構造: {selected_structure}")
    else:
        st.write("選択された機能がありません。")

else:
    # Inform the user if the data is not available
    st.write("データが見つかりません。他のページで収束的思考の構造テーブルを完成させてください。")


# DISPLAY THE REAGENT FEATURE IMPORTANCE OF THE FINAL DATA TO ALLOW THE USERS UNDERSTAND WHAT IS THE LLM (CHAT GPT IN THIS CASE) TAKING INTO ACCOUNT TO GENERATE THOSE SOLUTIONS

# We define a condition to run the Reagent method only when one of the selections has been selected
selections = [selected_function, selected_behavior, selected_structure]
non_empty_selections = []
for s in selections:
    if s:
        non_empty_selections.append(s)

if st.button("選択項目の特徴重要度を表示"):
        
    if len(non_empty_selections) == 1:
        # Define the type of input
        selected_item = non_empty_selections[0]
        
        # Determine the type of selection
        if selected_item == selected_function:
            type_of_input = "機能的"
            selected_item = selected_function
        elif selected_item == selected_behavior:
            type_of_input = "振る舞い的"
            selected_item = selected_behavior
        elif selected_item == selected_structure:
            type_of_input = "構造的"
            selected_item = selected_structure

        role_description = st.session_state["role_description_convergent"]
        design_problem = st.session_state["design_problem"]
        original_input = f"一言で言うと、{selected_item} は {design_problem} に対する {type_of_input} の解決策としてどれくらい優れていますか？回答は 'bad'、'poor'、'regular'、'good'、'great' のいずれかを使用してください。"

        # Generate unique keys for session state with the type of input and the selected item
        input_key = f"{type_of_input}_{selected_item}_original_input"
        tokens_key = f"{type_of_input}_{selected_item}_cleaned_tokens"
        scores_key = f"{type_of_input}_{selected_item}_token_scores_normalized"

        if input_key not in st.session_state or tokens_key not in st.session_state or scores_key not in st.session_state:
            # Calculate feature importance
            with st.spinner("特徴重要度を計算中です。しばらくお待ちください…"):
                original_first_token, cleaned_tokens, token_scores_normalized = calculate_feature_importance(original_input, role_description, tokenizer, model)
            # Save to session state
            st.session_state[input_key] = original_first_token
            st.session_state[tokens_key] = cleaned_tokens
            st.session_state[scores_key] = token_scores_normalized

            st.write("Score: ", st.session_state[input_key])
            html_output = visualize_scores(st.session_state[tokens_key], st.session_state[scores_key])
            st.markdown(html_output, unsafe_allow_html=True)
        else:
            # Retrieve and display existing data
            st.write("Score: ", st.session_state[input_key])
            html_output = visualize_scores(st.session_state[tokens_key], st.session_state[scores_key])
            st.markdown(html_output, unsafe_allow_html=True)

    elif len(non_empty_selections) == 0:
            st.warning("テーブルから少なくとも1つ、機能・振る舞い・構造のいずれかを選択してください。いずれも選択されていない場合、結果を計算することはできません。")

    else:
        st.warning("テーブルからは、機能・振る舞い・構造のいずれか1つのみを選択してください。複数選択された場合、結果を計算することはできません。")
