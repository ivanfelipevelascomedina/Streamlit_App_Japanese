# GUI FILTERING CODE

# IMPORT LIBRARIES
import streamlit as st

st.set_page_config(layout="wide")  # Set wide layout for the entire app

# DEFINE THE PAGE INITIAL INFORMATION

st.title("Information Page")

# Give user's information on how to use the APP
# Introduction
st.write("""
設計には創造性と正確さの両方が必要です。このアプリはそのプロセスを案内し、 
革新的なアイデアの創出を支援し、それらを効果的な解決策へと洗練します。このツールを最大限に活用するには、 
2つの重要な概念を理解することが不可欠です：**発散的および収束的思考** と **機能–振る舞い–構造（FBS）オントロジー**。
""")

# Divergent and Convergent Thinking
st.header("発散的および収束的思考")
st.write("""
デザインプロセスは、2つの思考モードを交互に繰り返します：

- **発散的思考**：ここでは、デザイナーが制限なく幅広いアイデアを生み出し、 
  創造性と革新性を促進します。
- **収束的思考**：ここでは、デザイナーがそれらのアイデアを洗練・評価し、最も実行可能で効果的な解決策を見極めます。

デザインプロセスは、循環的な旅だと考えてください。各イテレーションでは、発散的思考と収束的思考の組み合わせが行われ、まずは大胆に夢を描き（発散的思考）、その後で最良の成果を選びます（収束的思考）。このようなイテレーションを何度も繰り返すことで、最初の問題から最終的な解決策へと導くことができます。
""")

# Example: Chair Design
st.subheader("例：椅子の設計")
st.write("""
椅子の設計を想像してみてください：
- **発散的思考**：リサイクル素材で作られた椅子、空気で膨らむ椅子、マッサージ機能付きの椅子など、 
  何十ものアイデアをブレインストーミングするかもしれません。
- **収束的思考**：それらのアイデアの中から、実現可能性、コスト効率、機能性を評価し、 
  最良のデザインへと絞り込みます。
""")

# Add an explainatory image
st.image("images/Convergent_and_Divergent_Thinking.png", caption="Divergent and Convergent thinking for a chair design", use_container_width=True)

# FBS Ontology
st.header("機能–振る舞い–構造（FBS）オントロジーy")
st.write("""
設計は、**機能–振る舞い–構造（FBS）**フレームワークを通じて理解することもできます。これは、設計を3つの層に分解するものです：

- **機能（Function）**：その設計は何を達成しようとしているのか？（目的やテレオロジー）
- **振る舞い（Behavior）**：その設計はどのように機能するのか？（動作や効果）
- **構造（Structure）**：その設計は何で構成されているのか？（構成要素とその関係）

一般的な設計には複数の機能が含まれており、それぞれの機能は複数の振る舞いによって支えられ、多くの構造的要素が協調して働きます。FBSフレームワークは、この複雑さを管理可能な部分に分割することで、設計の分析・洗練・改善を容易にします。
""")

# Example: Chair Design
st.subheader("例：椅子の設計")
st.write("""
椅子の設計を想像してみてください：
- **機能（Function）**：椅子の機能は座るための場所を提供することです。
- **振る舞い（Behavior）**：椅子は人の体重を支え、快適さを提供します。
- **構造（Structure）**：椅子は、木材、金属、布などの素材でできた脚、座面、背もたれで構成されています。
""")

# Add an explainatory image
st.image("images/Simplified_function_behavior_structure.png", caption="Jiao, J., Pang, S., Chu, J., Jing, Y., and Zhao, T., 2021, An Improved FFIP Method Based on Mathematical Logic and SysML, Applied Sciences, 11(8), p. 3534. [Online]. Available: http://dx.doi.org/10.3390/app11083534.", width=400)

# How the App Helps
st.header("このアプリの使い方")
st.write("""
このアプリケーションは、設計プロセスの初期段階に焦点を当てています。この段階では、多様なアイデアを探求し、有望なものを特定することが目的です。このアプリは、機能–振る舞い–構造（FBS）オントロジーを一括で生成することで、機能・振る舞い・構造の複数の可能性を最初から視覚化できるようにサポートします。発散的思考を用いて創造的な解決策を探り、収束的思考を使って最適な選択肢を評価・選定することができます。
設計の後半段階では反復的な洗練が重要になることが多いですが、現在は最初の段階からの創造性と意思決定に焦点を当てています。これにより、将来的に必要であれば繰り返しに向けた確かな基盤を提供し、探求が鍵となるブレインストーミングや概念設計に最適なアプローチを実現します。

次の手順に従ってください：

1. **メインページ**：取り組む設計課題を定義し、Enterキーを押します。  
2. **発散的思考ページ**：設計課題に対して適切と思われる追加要件を選択し、それに基づいてFBSオントロジーを生成します。重要なのは、基本的な要件は既に初期設計課題で定義されているため、設計者が希望しない場合には追加要件を選択する必要はないということです。追加要件の選択は設計者の自由です。  
3. **発散的思考フィルタリングページ**：LLMが定義された設計課題および選択された要件に基づいて、すべての解決策間の関係性をどのように構築しているかを視覚化し、有用な可能性を評価します。  
4. **収束的思考ページ**：フィルタリングページで確認した情報をもとに、最も適切と思われる解決策を選び、必要に応じて手動で他の選択肢も追加します。FBSオントロジーの各要素に対して、5〜7件を上限に選定することを推奨します。  
5. **収束的思考フィルタリング結果ページ**：LLMが各選択肢をどのように評価しているか、また与えられた問題に対して最終的な設計案がどれほど優れているかを判断するために考慮している要素を視覚化します。
""")

# Add an explainatory image
st.image("images/Design_process_workflow.png", caption="Workflow Overview", use_container_width=True)
