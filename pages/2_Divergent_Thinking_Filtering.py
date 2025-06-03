# GUI FILTERING CODE

# IMPORT LIBRARIES
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from XAI_APP_utils import generate_embeddings, reduce_and_cluster, plot_interactive_clusters, rank_by_similarity, normalize_embeddings, enrich_with_wordnet

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

st.title("フィルタリング")

st.markdown("""
<div style="color: #1E90FF; font-size:16px;">
    <h5>重要：ワークフローに関する免責事項</h5>
    <b>発散的思考フィルタリング</b>ページでは、最も関連性の高い選択肢を視覚化、分析、および選択することができます。<br>
    <b>発散的思考</b>ページで解決策が生成されていない場合、このページは意図したとおりに機能しません。<br>
    続行する前に、解決策が存在することを確認してください。<br>
</div>
""", unsafe_allow_html=True)

st.markdown("""
## 使用方法：

1. **可視化結果を生成する**:  
   - このページを最初に開いたときは、可視化結果の計算に少し時間がかかります。ただし、次回以降は直接アクセスできるようになります。

2. **可視化モードを選択する**:  
   - ドロップダウンリストから可視化モードを選択してください。  
   - モードを選択すると、その可視化と各モードの説明が表示されます。

""")

# MAIN CODE OF THE FILTERING PART, HERE THE GPT-GENERATED FBS SOLUTIONS ARE VISUALIZED IN A 3D SPACE TO HELP THE DESIGNER UNDERSTAND THEM AND CHOOSE THE BEST ONES ACCORDING TO HIS OWN CRITERIA. HERE DESIGNERS ARE ALSO ALLOWED TO INCLUDE THEIR OWN FBS SOLUTIONS INTO THE DESIGN CYCLE
# Check if the OpenAI client, the design problem and the GPT-generated FBS solutions exist in session state
if "client" in st.session_state and "design_problem" in st.session_state and "fbs_table" in st.session_state and "selected_requirements" in st.session_state:

    # Retrieve the FBS data from session state and assure there are no empty entries (some could have been created during the data display in the divergent thinking)
    # Check if the lists are already calculated
    if (
        "functions_list" not in st.session_state or
        "behaviors_list" not in st.session_state or
        "structures_list" not in st.session_state
    ): 
        fbs_categories = ["Functions", "Behaviors", "Structures"]
        fbs_lists = {}

        for category in fbs_categories:
            # Clean the entries in each category, removing empty strings and non-strings
            fbs_lists[category] = [
                entry for entry in st.session_state.fbs_table[category]
                if isinstance(entry, str) and entry.strip()
            ]

        # Define the FBS lists
        functions_list = fbs_lists["Functions"]
        behaviors_list = fbs_lists["Behaviors"]
        structures_list = fbs_lists["Structures"]
        # Save cleaned lists in session state
        st.session_state["functions_list"] = functions_list
        st.session_state["behaviors_list"] = behaviors_list
        st.session_state["structures_list"] = structures_list
        # Define the FBS lists with WordNet enrichment
        #functions_list_enriched = enrich_with_wordnet(fbs_lists["Functions"])
        #behaviors_list_enriched = enrich_with_wordnet(fbs_lists["Behaviors"])
        #structures_list_enriched = enrich_with_wordnet(fbs_lists["Structures"])
        # Save enriched lists in session state
        #st.session_state["functions_list_enriched"] = functions_list_enriched
        #st.session_state["behaviors_list_enriched"] = behaviors_list_enriched
        #st.session_state["structures_list_enriched"] = structures_list_enriched

    # Check if embeddings and clustering results are already calculated
    if (
        "functions_embeddings" not in st.session_state or
        "behaviors_embeddings" not in st.session_state or
        "structures_embeddings" not in st.session_state or
        "functions_umap" not in st.session_state or
        "behaviors_umap" not in st.session_state or
        "structures_umap" not in st.session_state
    ):        
        # Generate embeddings
        st.write("Preparing functions visualization")
        functions_embeddings = generate_embeddings(functions_list)
        st.write("Preparing behaviors visualization")
        behaviors_embeddings = generate_embeddings(behaviors_list)
        st.write("Preparing structures visualization")
        structures_embeddings = generate_embeddings(structures_list)

        # Save embeddings in session state
        st.session_state["functions_embeddings"] = functions_embeddings
        st.session_state["behaviors_embeddings"] = behaviors_embeddings
        st.session_state["structures_embeddings"] = structures_embeddings

        # Dimensionality reduction and clustering
        functions_umap, functions_clusters, functions_silhouette, functions_ch = reduce_and_cluster(functions_embeddings)
        st.session_state["functions_umap"] = functions_umap
        st.session_state["functions_clusters"] = functions_clusters
        st.session_state["functions_silhouette"] = functions_silhouette
        st.session_state["functions_ch"] = functions_ch

        behaviors_umap, behaviors_clusters, behaviors_silhouette, behaviors_ch = reduce_and_cluster(behaviors_embeddings)
        st.session_state["behaviors_umap"] = behaviors_umap
        st.session_state["behaviors_clusters"] = behaviors_clusters
        st.session_state["behaviors_silhouette"] = behaviors_silhouette
        st.session_state["behaviors_ch"] = behaviors_ch

        structures_umap, structures_clusters, structures_silhouette, structures_ch = reduce_and_cluster(structures_embeddings)
        st.session_state["structures_umap"] = structures_umap
        st.session_state["structures_clusters"] = structures_clusters
        st.session_state["structures_silhouette"] = structures_silhouette
        st.session_state["structures_ch"] = structures_ch
    #else:
    #    st.write("Visualization preparation complete")

    # Give the user the option to select what kind of output to display to better understand the generated solutions
    # Initialize session state for the selectbox if not already set
    if "selected_option" not in st.session_state:
        st.session_state.selected_option = "Display All the Solutions"  # Default value

    # Create a selectbox tied to session state using the `key` parameter
    st.selectbox(
        "オプションを選択してください:",
        ["Display All the Solutions", "Solution Space Visualization", "Order Solutions by Similarity to the Design Problem", "Order Solutions by Similarity to a Given Requirement", "Display Each Solution's Most Similar Requirement"],
        key="selected_option",  # Automatically syncs with session_state
        help="生成された解決策を検討するために、分析または可視化のオプションを選択してください。"
    )

    if st.session_state.selected_option == "Display All the Solutions":
        # Visualize the generated data
        st.write("### AIが生成したすべての解決策を可視化中")

        with st.expander("「すべての解決策を表示」とは何ですか？"):
            st.write("""
            このオプションを使用すると、前の設計ステップ（発散的思考）でうまく可視化できなかった場合に、AIが生成したすべての解決策を再度可視化することができます。
         """)

        # Visualize all the solutions
        st.dataframe(st.session_state.fbs_table, height=400, use_container_width=True)

    elif st.session_state.selected_option == "Solution Space Visualization":
        # Visualize the embeddings
        st.write("### 解決策空間の可視化")

        # Add a short explanation of the selected option
        with st.expander("「解決策空間の可視化」とは何ですか？"):
            st.write("""
            このオプションでは、生成されたFBSの解決策（機能、振る舞い、構造）を、テキスト同士の関係性を示すインタラクティブな2次元空間に表示します。  
            各点は1つの解決策を表しており、点と点の距離が近いほど、アイデアが類似していることを意味します。  
            この可視化を利用して、類似した解決策を見つけたり、ユニークなアイデアを特定したり、提案の多様性を探索したりすることができます。
            """)

        # Display the embedding space for each of the FBS solution spaces
        fbs_categories = ["Functions", "Behaviors", "Structures"]
        for category in fbs_categories:
            # Retrieve data from session state
            umap_embeddings = st.session_state[f"{category.lower()}_umap"]
            clusters = st.session_state[f"{category.lower()}_clusters"]
            data_list = st.session_state[f"{category.lower()}_list"]  # Use cleaned list
            silhouette_score = st.session_state[f"{category.lower()}_silhouette"]
            ch_index = st.session_state[f"{category.lower()}_ch"]

            # Plot interactive clusters
            #st.write(f"#### {category} Space:")
            plot_interactive_clusters(
                umap_embeddings,
                clusters,
                data_list,
                f"{category} Space"
            )
            #st.write("Silhouette: ", silhouette_score)
            #st.write("CH: ", ch_index)

    elif st.session_state.selected_option == "Order Solutions by Similarity to the Design Problem":
        # Order solutions based on their similarity to the given design problem
        st.write("### デザイン課題への類似度による解決策のランキング")

        # Add a short explanation of the selected option
        with st.expander("「デザイン課題への類似度によるランキング」とは何ですか？"):
            st.write("""
            このオプションでは、各解決策がデザイン課題にどれだけ適合しているかに基づいて順位付けを行います。  
            順位付けにはコサイン類似度スコアを使用しており、各解決策が設計目標とどれだけ近いかを数値的に評価します。  
            これにより、最も関連性の高いアイデアを素早く特定するのに役立ちます。
            """)

        # Generate embedding for the design problem
        design_problem_embedding = generate_embeddings([st.session_state.design_problem])[0]
        design_problem_embedding = normalize_embeddings([design_problem_embedding])[0]

        fbs_categories = ["Functions", "Behaviors", "Structures"]
        for category in fbs_categories:
            st.write(f"#### Ranking {category}:")
            embeddings = st.session_state[f"{category.lower()}_embeddings"]
            data_list = st.session_state[f"{category.lower()}_list"]  # Use cleaned list
            rank_by_similarity(design_problem_embedding, embeddings, data_list)
        
    elif st.session_state.selected_option == "Order Solutions by Similarity to a Given Requirement":
        # Order solutions based on their similarity with a chosen requirement
        st.write("### 特定の要件への類似度による解決策のランキング")

        # Add a short explanation of the selected option
        with st.expander("「特定の要件への類似度によるランキング」とは何ですか？"):
            st.write("""
            このオプションでは、特定の要件を選択し、その要件との類似度に基づいてFBSの解決策を順位付けすることができます。  
            これにより、各解決策が個々の要件をどの程度満たしているかを把握するのに役立ちます。
            """)

        # Select a requirement
        selected_requirement = st.selectbox(
            "要件を選択してください:",
            st.session_state.selected_requirements
        )

        if selected_requirement:
            # Generate embedding for the selected requirement
            requirement_embedding = generate_embeddings([selected_requirement])[0]
            requirement_embedding = normalize_embeddings([requirement_embedding])[0]

            fbs_categories = ["Functions", "Behaviors", "Structures"]
            for category in fbs_categories:
                st.write(f"#### 要件に対する{category}のランキング: {selected_requirement}")
                embeddings = st.session_state[f"{category.lower()}_embeddings"]
                data_list = st.session_state[f"{category.lower()}_list"]  # Use cleaned list
                rank_by_similarity(requirement_embedding, embeddings, data_list)

    elif st.session_state.selected_option == "Display Each Solution's Most Similar Requirement":
        # Display each solution's most similar requirement
        st.write("### 各解決策に最も類似した要件を表示")

        # Add a short explanation of the selected option
        with st.expander("「各解決策に最も類似した要件を表示」とは何ですか？"):
            st.write("""
            このオプションでは、コサイン類似度を用いて、各FBS解決策に最も類似した要件を自動的にマッチングします。  
            これにより、各解決策が選択された要件全体とどのように整合しているかを把握するのに役立ちます。
            """)

        # Generate embeddings for all requirements
        requirement_embeddings = generate_embeddings(st.session_state.selected_requirements)
        requirement_embeddings = normalize_embeddings(requirement_embeddings)

        fbs_categories = ["Functions", "Behaviors", "Structures"]
        for category in fbs_categories:
            st.write(f"#### {category}に最も類似した要件:")
            embeddings = st.session_state[f"{category.lower()}_embeddings"]
            data_list = st.session_state[f"{category.lower()}_list"]  # Use cleaned list

            most_similar_requirements = []
            for i, solution_embedding in enumerate(embeddings):
                similarities = cosine_similarity([solution_embedding], requirement_embeddings)[0]
                most_similar_index = similarities.argmax()
                most_similar_requirement = st.session_state.selected_requirements[most_similar_index]
                most_similar_requirements.append((data_list[i], most_similar_requirement, similarities[most_similar_index]))

            # Display the table
            df = pd.DataFrame(most_similar_requirements, columns=["Solution", "Most Similar Requirement", "Similarity"])
            st.dataframe(df, use_container_width=True)
        
        else:
            st.write("可視化のオプションを選択してください")


    # Add a reset button outside the try-except block
    if st.button("Reset", help="ここをクリックすると、フィルタリングプロセスがリセットされ、最初からやり直すことができます。  ただし、ここをクリックするとすべてのデータが失われるため、ご注意ください。"):
        for key in st.session_state.keys():
            del st.session_state[key]

else:
    st.error("デザイン課題を入力し、OpenAIクライアントがメインページで初期化されていることを確認してください。")