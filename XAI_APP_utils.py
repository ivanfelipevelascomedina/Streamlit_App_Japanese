# THIS CODE INCLUDES THE DIFFERENT FUNCTIONS THAT WE WILL BE USING IN OTHER PAGES

# IMPORT LIBRARIES
import streamlit as st
import re
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from openai import OpenAI
from transformers import RobertaTokenizer, RobertaForMaskedLM
import random
random.seed(42) # Introduce a seed to reduce variability
import torch
np.random.seed(42) # Itroduce a seed to reduce variability
import math
from IPython.display import HTML, display
import unicodedata
import json
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances


# DEFINE THE FUNCTIONS TO USE

# 1. DIVERGENT THINKING FUNCTIONS
    
# This function generates Chat o1 answers for a given design problem and FBS ontology element
def generate_design_output(design_problem, ontology_element, role_description, ontology_element_definition, ontology_element_example):
    prompt = (
        #f"Design Goal: '{design_problem}'\n"
        #    f"Your task is to propose as many distinct and creative '{ontology_element}' as possible for the given Design Goal.\n"
        #    f"- Focus on generating diverse '{ontology_element}' that vary in approach, scope, and perspective.\n" 
        #    f"- Include a mix of conventional and unconventional ideas, ensuring broad representation across different possibilities.\n"
        #    f"- Do not limit the solutions to feasibility—some may be abstract or exploratory in nature.\n"
        #    f"Provide all the solutions as a single list, separated by commas, without any additional explanation or numbering.\n" 
        #    f"Diversity and uniqueness are the most important criteria, followed by quantity and relevance to the problem.\n"
        #    f"--- END OF TASK ---\n"
        f"設計目標: '{design_problem}'\n"
            f"あなたのタスクは、与えられた設計目標に対して、できるだけ多くの多様で創造的な{ontology_element}を提案することです。\n"
            f"- '{ontology_element_definition}'\n"
            f"- '例えば、ターボチャージャーにおける{ontology_element}には、次のような要素があります: {ontology_element_example}'\n"
            f"- 異なる可能性を探る多様な{ontology_element}を生成してください。\n"
            f"- 各{ontology_element}がそのカテゴリに正しく対応していることを確認してください。\n"
            f"- 実現可能性に制限せず、抽象的または探索的な案も含めて構いません。\n"
            f"- すべての解答は日本語で構いませんが、追加の説明や番号なしで**英語のカンマ（,）**で区切った単一リストとして提示してください。日本語の読点（、）を使用すると、プログラムが正しく動作しません。\n"
            f"正確性と多様性が最も重要な評価基準であり、それに次いで量と関連性が重視されます。\n"
            f"--- タスク終了 ---\n"
    )
    # Check if the openai client can be accessed
    if "client" not in st.session_state:
        st.error("OpenAIクライアントが初期化されていません。有効なAPIキーを入力してください。")
        return None

    # Generate an answer if the openai client can be access
    try:
        response = st.session_state.client.chat.completions.create(
            model="o1-mini", # Since all the design process relies on this stage we will use the latest model in its mini version to assure also fast replies and a good UX
            messages=[
                #{"role": "system", "content": role_description},
                {"role": "user", "content": prompt}
            ]
        )

        # Validate response structure
        if not hasattr(response, "choices") or not response.choices:
            st.warning("APIのレスポンスに有効な選択肢が含まれていませんでした。")
            return []

        # Extract the response content to prepare it before returning it in the function call
        content = response.choices[0].message.content.strip()

        # Check if the content is empty or improperly formatted
        if not content:
            st.warning("レスポンスが空でした。")
            return []
        if "," not in content:
            st.warning(f"予期しないレスポンス形式です。: {content}")
            return []

        # Process the response into a list of solutions for the future data handling
        solutions = []
        # 1: Split the content by commas
        raw_solutions = content.split(',')
        # 2: Trim whitespace from each solution and filter out empty ones
        for item in raw_solutions:
            cleaned_item = item.strip()  # Remove extra spaces
            cleaned_item = re.sub(r"^\d+[.)-]?\s*", "", cleaned_item) # Remove numbered prefixes like "1.", "1 ", "1)", "1-" etc.
            cleaned_item = re.sub(r"^\s*[-•]\s*", "", cleaned_item)  # Remove leading hyphens or bullets
            cleaned_item = cleaned_item.lower()  # Convert to lowercase for consistency
            if cleaned_item:  # Keep only non-empty items
                solutions.append(cleaned_item)
        # Step 3: Return the list of cleaned solutions
        return solutions

    except Exception as e:
        st.error(f"レスポンスの生成中にエラーが発生しました。: {e}")
        return None
    

# 2.0 FILTERING FUNCTIONS

# This function creates embeddings for a text list using openai API fast embedding model "ext-embedding-3-small" to assure quick results
def generate_embeddings(text_list, model="text-embedding-3-small"):
    embeddings = []
    progress_bar = st.progress(0)  # Initialize progress bar
    for i, text in enumerate(text_list):
        try:
            response = st.session_state.client.embeddings.create(
                model=model,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            embeddings.append(None)
            st.warning(f"Error generating embedding for '{text}': {e}")
        progress_bar.progress((i + 1) / len(text_list))  # Update progress
    progress_bar.empty()  # Clear progress bar
    return [e for e in embeddings if e is not None]

# This function generates conceptualized terms for a given list to increase the semantic space and improve clustering creation
def enrich_with_wordnet(data_list):
    enriched_list = []
    # Iterate through the data list, split each of the entries and generate hypernyms and synonums for each of thme
    for entry in data_list:
        conceptualized = []
        for word in entry.split():
            synsets = wn.synsets(word)
            if synsets:
                # Add hypernyms and synonyms
                hypernyms = synsets[0].hypernyms()
                related_terms = [lemma.name().replace('_', ' ') for h in hypernyms for lemma in h.lemmas()]
                conceptualized.extend(related_terms)
        conceptualized.append(entry)  # Include the original entry
        enriched_list.append(" ".join(set(conceptualized)))  # Remove duplicates
    return enriched_list

# This function reduces embeddings dimensions and creates density based clusters based on them
def reduce_and_cluster(embeddings):
    # Log the dimensionality reduction and clustering process
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)

    # Apply PCA
    max_components = min(normalized_embeddings.shape[1], normalized_embeddings.shape[0] - 1)  # Ensure it does not exceed the embedding dimensions
    pca = PCA(n_components=max_components)
    reduced_embeddings = pca.fit_transform(normalized_embeddings)
    
    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, metric="cosine", random_state=42, n_components=3) # Since we are working with small noise datasets this parameters ensure a useful visualization for the users
    umap_embeddings = reducer.fit_transform(reduced_embeddings)

    # Precompute cosine distances for HDBSCAN
    #distance_matrix = cosine_distances(umap_embeddings).astype(np.float64)
    
    # Cluster using HDBSCAN
    #clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric="precomputed")
    #clusters = clusterer.fit_predict(distance_matrix)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric="euclidean") # Since we are working with small noisy datasets this parameters ensure a useful visualization for the users. Eventhough, a prior, it may seem that cosine would work better than euclidean, after trying it it seems euclidean is a better choice
    clusters = clusterer.fit_predict(umap_embeddings)

    # Evaluate clustering
    noise_filter = clusters != -1
    silhouette_avg, ch_index = None, None
    if np.sum(noise_filter) > 1:
        valid_embeddings = umap_embeddings[noise_filter]
        valid_clusters = clusters[noise_filter]
        silhouette_avg = silhouette_score(valid_embeddings, valid_clusters)
        ch_index = calinski_harabasz_score(valid_embeddings, valid_clusters)

    return umap_embeddings, clusters, silhouette_avg, ch_index

# This function creates an interactive embedding space to allow the user visualize each of the LLM generated solutions
def plot_interactive_clusters(umap_embeddings, clusters, labels, title):
    # Check the dimensionality of the embeddings
    dimensions = umap_embeddings.shape[1]
    
    # Create scatter plot based on dimensionality
    if dimensions == 2:
        fig = px.scatter(
            x=umap_embeddings[:, 0],
            y=umap_embeddings[:, 1],
            color=clusters.astype(str),
            hover_data={"Text": labels},
            title=title,
            opacity=0.5  # Adjust transparency (0 = fully transparent, 1 = fully opaque)
        )
    elif dimensions == 3:
        fig = px.scatter_3d(
            x=umap_embeddings[:, 0],
            y=umap_embeddings[:, 1],
            z=umap_embeddings[:, 2],
            color=clusters.astype(str),
            hover_data={"Text": labels},
            title=title,
            opacity=0.5  # Adjust transparency (0 = fully transparent, 1 = fully opaque)
        )
    else:
        raise ValueError("Unsupported number of dimensions for UMAP embeddings.")
    
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# this function ranks the solutions by the similarity their embeddings have with a given one (design goal or a specific requirement)
def rank_by_similarity(reference_embedding, embeddings, labels):
    similarities = cosine_similarity([reference_embedding], embeddings)[0]
    ranked = sorted(zip(labels, similarities), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(ranked, columns=["Solution", "Similarity"])
    st.dataframe(df, use_container_width=True)  # Display ranked table
    return df

# This function normalizes embeddings to assure consistency when calculating distances or similarities
def normalize_embeddings(embeddings):
    return [embedding / np.linalg.norm(embedding) for embedding in embeddings]
  

# 3. CONVERGENT THINKING FUNCTIONS

# 4. FINAL RESULTS FUNCTIONS
    
# This function generates a Chat GPT4 answer and the first 5 logprobs for a given input
def answer_generation(input, role_description):

    # Check if the openai client can be accessed
    if "client" not in st.session_state:
        st.error("OpenAI client is not initialized. Please provide a valid API key.")
        return None

    # Generate an answer if the openai client can be access
    try:
        response = st.session_state.client.chat.completions.create(
            model="gpt-4o-mini", # Since the design process is almost done and we are just using this stage as a tool to understand the results we will use a fast model such as 4o-mini to ensure fast results
            messages=[
                {"role": "system", "content": role_description},
                {"role": "user", "content": input},
            ],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=5,
        )
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# This function takes an output with logprobs values from ChatGPT4 and returns the ordered values of probs and token (is used to handle data easier and to work with probs instead of logprobs as expected in the ReAgent method)
def extract_probs_information(response):
    """
    Extract tokens and probabilities from the `logprobs` attribute of the ChatCompletion response.
    
    Args:
        response (ChatCompletion): The response object returned by the `answer_generation` function.

    Returns:
        list of list: A list where each element corresponds to the top alternatives for a token.
                      Each element is a list of dictionaries with `token` and `prob`.
    """
    # Access logprobs content
    logprobs_data = response.choices[0].logprobs.content

    top_alternatives = []  # Store all token probabilities
    prob_sum = 0  # Probability sum tracker

    # Iterate through each token's logprobs
    for logprob_entry in logprobs_data:
        # Convert top logprobs to a list of token-probability pairs
        alternatives = [
            {"token": top_logprob.token, "prob": math.exp(top_logprob.logprob)}
            for top_logprob in logprob_entry.top_logprobs
        ]
        # Add to the results
        top_alternatives.append(alternatives)

        # Update the probability sum
        for alt in alternatives:
            prob_sum += alt["prob"]

        # Break if probabilities sum to 1 (handling floating-point precision)
        if abs(prob_sum) == 1:
            break

    return top_alternatives

# This function subtitute a defined percentage of the tokens randomly of a given text with RoBERTa predictions
def substitute_tokens(input_ids, num_tokens, replace_ratio, tokenizer, model):

    # Clone input_ids to avoid modifying the original tensor
    input_ids_for_modification = input_ids.clone()
    
    num_tokens_to_mask = max(1, int(num_tokens * replace_ratio)) # Define the number of tokens to mask (the ones that will be substituted with RoBERTa generated solutions) according to the replace ratio

    # Randomly select token indices to mask
    mask_indices = random.sample(range(0, num_tokens), num_tokens_to_mask)
    
    # Replace selected tokens with <mask>
    for idx in mask_indices:
        input_ids_for_modification[idx] = tokenizer.mask_token_id
    
    # Predict all masks simultaneously
    with torch.no_grad():
        outputs = model(input_ids_for_modification.unsqueeze(0))  # Add batch dimension
        logits = outputs.logits
    
    # Decode predictions for each <mask>
    for mask_idx in mask_indices:
        # Get the logits for the current <mask>
        mask_logits = logits[0, mask_idx, :]
        predicted_token_id = torch.argmax(mask_logits).item()
        
        # Replace the mask token with the predicted token
        input_ids_for_modification[mask_idx] = predicted_token_id
    
    # Decode the final sequence
    substituted_text = tokenizer.decode(input_ids_for_modification, skip_special_tokens=True)
    
    return substituted_text, mask_indices

# This function takes to lists of probs and checks the probability change for the first token of the first list
def calculate_prob_difference(original_probs, modified_probs):
    
    # Get the original token and its prob
    original_token = original_probs[0][0]['token'].lower() # Set to lower cases to avoid comparison issues
    original_token_prob = original_probs[0][0]['prob']

    # Look for the original token in the modified probs
    modified_prob = 0.0  # Default to 0 if the token is not found
    for entry in modified_probs[0]:
        if entry['token'].lower() == original_token: # Set to lower cases to avoid comparison issues
            modified_prob = entry['prob']
            break

    # Calculate the prob difference
    prob_diff = original_token_prob - modified_prob
    
    return prob_diff

def visualize_scores(tokens, normalized_scores):
    max_score = max(normalized_scores)
    min_score = min(normalized_scores)  # To map scores effectively
    score_range = max_score - min_score

    # HTML for the color legend with a purple gradient
    legend_html = """
    <div style="margin-bottom: 10px;">
        <span style="display: inline-block; width: 20px; height: 20px; background-color: rgb(240, 240, 255);"></span> Low Importance
        <span style="display: inline-block; width: 20px; height: 20px; background-color: rgb(180, 150, 255); margin-left: 10px;"></span> Medium Importance
        <span style="display: inline-block; width: 20px; height: 20px; background-color: rgb(128, 0, 128); margin-left: 10px;"></span> High Importance
    </div>
    """

    # Token visualization
    token_html = ""
    for token, normalized_score in zip(tokens, normalized_scores):
        # Normalize the score to a 0–1 range for consistent mapping
        scaled_score = (normalized_score - min_score) / score_range if score_range > 0 else 0

        # Create a gradient: low scores are light purple, high scores are dark purple
        r = int(240 - (112 * scaled_score))  # Red intensity decreases
        g = int(240 - (240 * scaled_score))  # Green intensity decreases
        b = int(255 - (127 * scaled_score))  # Blue intensity decreases slightly

        # Create the color string
        color = f"rgb({r}, {g}, {b})"

        # Add token with a colored background
        token_html += f'<span style="background-color: {color}; padding:2px;">{token}</span> '
        
    # Combine legend and token visualization
    html_output = legend_html + token_html

    return html_output

# This function cleans up LLM-generated tokens for visualization purposes
def clean_tokens(tokens):
    cleaned_tokens = []
    for token in tokens:
        token = unicodedata.normalize("NFKC", token) # Normalize the token to handle any encoding issues
        token = token.strip() # Strip any surrounding whitespace
        token = token.replace("Ċ", " ") # Clean up blank space
        token = token.replace("Ġ", "") # Clean up word beginnings
        cleaned_tokens.append(token)
        
    return cleaned_tokens

# This function checks if the stopping condition for the token importance calculation is met
def calculate_stopping_condition(input_ids, num_tokens, replace_ratio, token_scores, original_probs, role_description, tokenizer, model):
    # Set the stopping condition as false
    stopping_condition = False

    # Clone input_ids to avoid modifying the original tensor
    input_ids_for_stopping = input_ids.clone()

    # Calculate the number of tokens to replace (70% of the sequence)
    num_tokens_to_mask = int(replace_ratio * (num_tokens-2)) # We substract 2 to avoid <s> and </s>

    # Get indices of the 70% least important tokens
    mask_indices = np.argsort(token_scores)[:num_tokens_to_mask]
    
    # Replace selected tokens with <mask>
    for idx in mask_indices:
        input_ids_for_stopping[idx] = tokenizer.mask_token_id
    
    # Predict all masks simultaneously
    with torch.no_grad():
        outputs = model(input_ids_for_stopping.unsqueeze(0))  # Add batch dimension
        logits = outputs.logits
    
    # Decode predictions for each <mask>
    for mask_idx in mask_indices:
        # Get the logits for the current <mask>
        mask_logits = logits[0, mask_idx, :]
        predicted_token_id = torch.argmax(mask_logits).item()
        
        # Replace the mask token with the predicted token
        input_ids_for_stopping[mask_idx] = predicted_token_id
    
    # Decode the final sequence
    substituted_text = tokenizer.decode(input_ids_for_stopping, skip_special_tokens=True)
    #st.write("Substituted text: ", substituted_text)

    # Generate predictions for the modified input
    modified_probs = extract_probs_information(answer_generation(substituted_text, role_description))

    # Check if the original target token is in the top-3 predictions
    original_token = original_probs[0][0]['token'].lower()
    top_3_tokens = [] # Initialize an empty list to store the top-3 tokens
    for entry in modified_probs[0][:3]: # Loop through the first 3 entries in modified_probs[0]
        top_3_tokens.append(entry['token'].lower()) # Convert the token to lowercase and add it to the list
    #st.write(f"Top-3 tokens: {top_3_tokens}")

    if original_token in top_3_tokens:
        stopping_condition = True

    return stopping_condition

# This is the main function
def calculate_feature_importance(original_input, role_description, tokenizer, model):
    # Define the main parameters
    replace_ratio = 0.3 # Percentage of tokens that will be replaced in each iteration for the token importance calculation
    replace_ratio_stopping_condition = 0.7 # Percentage of non important tokens replaced while checking the stopping condition

    # Calculate the most probable answers for the given input with Chat GPT4, this will allow us to compare how much each of the input tokens affects the generated answer. Additionally we assure that the given answer by Chat GPT4 is one of the Likert-type scale ones
    original_probs = extract_probs_information(answer_generation(original_input, role_description))
    original_first_token = original_probs[0][0]['token'].lower() # Extract the value of the first output token to see if it belong to the Likert-typer scale categories and convert it to minus to compare
    #st.write("Original input: ", original_input)
    #st.write("Original first token probability: ", original_first_token)
    
    # Asure that the first output token from the original input follows the descrived rating criteria in the role description
    for l in range(10): # Try to generate a Chat GPT4 answer 10 times, if not we say that the classification was not possible    
        if original_first_token not in ("excellent", "good", "regular", "poor", "bad"): # If the most probable output token is not one of the ones defined for the Likert-type scale we indicate ChatGPT4 to do so. This words have been checked to be single tokens for GPT4 and GPT4-mini tokenizer
            if l == 0: # In the first iteration we indicate Chat GPT4 that the classification was not done correctly, this will be added to the input for future iterations
                input = (original_input + "\n\nIMPORTANT: Your previous response did not provide a classification based on the following categories: excellent, good, regular, bad, awful. This time, ensure it aligns with the specified criteria.")
            original_probs = extract_probs_information(answer_generation(input, role_description))
            original_first_token = original_probs[0][0]['token'].lower()
        else: # Finish the loop if the answer is generated accorfing to the given criteria
            break

    # Tokenize the given input with RoBERTa and measure its lenght in terms on tokens to define the number of necessary iterations to evaluate token importance
    original_tokens = tokenizer(original_input, return_tensors='pt') # Token values
    original_token_ids = original_tokens['input_ids'][0] # Token ids
    original_token_count = len(original_token_ids) - 2 # How many tokens we have and exclude <s> and </s>
    #st.write("Original tokens: ", original_tokens)

    # Handle possible errors due to long inputs
    if original_token_count > 512:
        raise ValueError("Input exceeds the maximum token limit of 512. Please shorten the input text.")
    
    # Define the common tokens in the input structure to give them a higher score and speed up the convergence process. Since our goal is to gain a better UX and we know the structure of the inputs we have, this approach can get us closer to that point.
    common_parts = [
        "In", "one", "word", "how", "good", "is", 
        "as", "a", "solution", "for", "?"
    ]
    common_token_ids = tokenizer(" ".join(common_parts), return_tensors='pt')['input_ids'][0]

    # Initialize scores with a bias for common tokens
    token_scores_logit = []
    scaled_probs_diff_vect = []
    for idx, token_id in enumerate(original_token_ids):
        if token_id in common_token_ids:  # Check if token is part of the common structure
            token_scores_logit.append(0.1)  # Assign a higher initial value (0.1)
        else:
            token_scores_logit.append(0)  # Assign the default value (0)
        scaled_probs_diff_vect.append(0)

    # Normalize token_scores into another list for the calculations
    token_scores_normalized = softmax(token_scores_logit)
        
    # Initialize a dictionary to store historical values
    historical_token_scores_logit = {}
    historical_token_scores_normalized = {}
    historical_token_scores_logit[0] = token_scores_logit.copy()  # Store the initial list of logit values
    historical_token_scores_normalized[0] = token_scores_normalized.copy()  # Store the initial list of normalized values

    # Create a list to store each iteration data and check it once the process is finished
    all_iterations_data = []  # List to store iteration data

    # Define the calculation loop conditions
    stop = False
    max_iterations = 30 # Maximum number of loop iterations
    min_iterations = 10 # Minimum number of loop iterations

    # Iterative importance evaluation until the stopping condition is met
    for i in range(max_iterations):
        iteration_data = {}  # Initialize a dictionary for the current iteration
        iteration_data["iteration"] = i
        #st.write(i)

        # 1. MODIFY THE ORIGINAL INPUT CHANGING "replace_ratio" % OF THE TOKENS
        
        # Modify the original input according to the replace ratio and obtain which tokens have been modified
        modified_input, replaced_indices = substitute_tokens(original_token_ids, original_token_count, replace_ratio, tokenizer, model)
        #st.write("Modified input: ",i, modified_input)
        
        # Save replaced indices
        iteration_data["original_input"] = original_input
        iteration_data["modified_input"] = modified_input
        iteration_data["replaced_indices"] = replaced_indices

        # 2. CALCULATE THE OUTPUT PROBS FOR THE NEW MODIFIED INPUT
        
        # Calculate probs for the modified input
        modified_probs = extract_probs_information(answer_generation(modified_input, role_description))

        #Save the probs information
        iteration_data["original_probs"] = original_probs
        iteration_data["modified_probs"] = modified_probs

        # 3. COMPARE THE NEW PROBS WITH THE OLD PROBS AND ASIGN A SCORE TO EACH OF THE TOKENS ACCORDING TO THAT

        # Step 3.1: Calculate the difference in the first output token probability
        prob_diff = calculate_prob_difference(original_probs, modified_probs)
        iteration_data["prob_diff"] = prob_diff

        # Step 3.2: Update importance scores for replaced tokens according to the formulas described in the ReAgent paper
        # Step 3.2.1 Scale the prob difference
        scaled_diff = prob_diff / original_token_count
        iteration_data["scaled_delta_p"] = scaled_diff
        # Step 3.2.2 Create the Scaled prob differences vector according to the ReAgent method  
        for idx in range(original_token_count): # Add scaled negative update for the non-replaced tokens
            if idx in replaced_indices:
                scaled_probs_diff_vect[idx] = scaled_diff # Add scaled positive value for the replaced tokens
            elif idx not in replaced_indices:
                scaled_probs_diff_vect[idx] = -scaled_diff # Add scaled negative value for the non-replaced tokens

        # Save scaled_probs_diff_vect and pre-softmax token scores
        iteration_data["scaled_probs_diff_vect"] = scaled_probs_diff_vect.copy()
        iteration_data["logit_scores_before_logit_update"] = token_scores_logit.copy()
                
        # Step 3.3: Calculate logit update
        for p in range(original_token_count):
            logit_term = (scaled_probs_diff_vect[p] + 1) / 2
            if (1 - logit_term) == 0: # Handle ZeroDivision Errors
                epsilon = 1e-10
                logit_value = math.log((logit_term+epsilon)/(1-logit_term+epsilon))
                if p in replaced_indices: # Ensure that token importance updates maintain the total values with no variation
                    logit_value = (1 - replace_ratio) * logit_value
                elif p not in replaced_indices:
                    logit_value = replace_ratio * logit_value 
            else:
                logit_value = math.log(logit_term/(1-logit_term))
                if p in replaced_indices:
                    logit_value = (1 - replace_ratio) * logit_value
                elif p not in replaced_indices:
                    logit_value = replace_ratio * logit_value 
            token_scores_logit[p] = historical_token_scores_logit[i][p] + logit_value # The updated token score is the token score from the last iteration plus the logit

        iteration_data["logit_scores_before_softmax"] = token_scores_logit.copy()
            
        # Step 3.5: Normalize scores using softmax
        token_scores_normalized = softmax(token_scores_logit)
        iteration_data["normalized_scores_after_softmax"] = token_scores_normalized.copy()
    
        # Update historical scores with this iteration values
        historical_token_scores_logit[i+1] = token_scores_logit.copy()
        historical_token_scores_normalized[i+1] = token_scores_normalized.copy()

        # 4. FINISH CALCULATING TOKEN SCORES AND CHECK IF THE STOPPING CONDITION IS MET

        all_iterations_data.append(iteration_data)  # Add the current iteration data to the list

        # Calculate convergence to check if the score calculation loop is ready or if the iteration should continue according to the defined stopping condition (ReAgent one or average tokens scores are below a certain threshold in the last defined minimum number of iterations)
        stop = calculate_stopping_condition(original_token_ids, original_token_count, replace_ratio_stopping_condition, token_scores_normalized, original_probs, role_description, tokenizer, model)
        if (i >  min_iterations) and (stop == True):
            #st.write("Convergence reached.")
            break
    
    # Handle unsupported types (Numpy Arrays)
    for entry in all_iterations_data: 
        for key, value in entry.items():
            if isinstance(value, np.ndarray):
                entry[key] = value.tolist()  # Convert NumPy array to list
    # Convert token IDs to human-readable tokens
    raw_tokens = tokenizer.convert_ids_to_tokens(original_token_ids.tolist())
    
    # Clean the tokens for visualization purposes
    cleaned_tokens = clean_tokens(raw_tokens)
    cleaned_tokens = [token for token in cleaned_tokens if token not in ["<s>", "</s>"]] # Remove first and last token to avoid visualization errors
    #st.write("Cleaned Tokens:", cleaned_tokens)

    extra_input = (f"You are a design expert skilled in critically evaluating design proposals. Your task is to classify each proposal as Bad, Poor, Regular, Good, or Excellent, based on its structural feasibility, functionality, innovation, efficiency, and relevance to the design goal.\n"
                        f"- Do not assume every proposal is excellent; critically assess both strengths and weaknesses.\n"  
                        f"- Bad (1/5): The solution is ineffective, impractical, or irrelevant to the design goal.\n"  
                        f"- Poor (2/5): The solution has major flaws, such as high cost, weak feasibility, or poor performance.\n"  
                        f"- Regular (3/5): The solution is functional but lacks uniqueness, optimization, or strong benefits.\n"  
                        f"- Good (4/5): The solution is well-designed, feasible, and useful, with only minor drawbacks.\n"  
                        f"- Excellent (5/5): The solution is highly innovative, fully feasible, and significantly improves the design goal.\n"  
                        f"Be objective, provide a **fair** evaluation, and ensure diversity in ratings based on the actual quality of the proposal.\n"
                    )

    strict_input = original_input + extra_input
    strict_probs = extract_probs_information(answer_generation(strict_input, role_description))
    strict_first_token = strict_probs[0][0]['token'].lower()
    # Asure that the first output token from the strict input follows the descrived rating criteria in the role description
    for o in range(10):
        if strict_first_token not in ("excellent", "good", "regular", "poor", "bad"):
            if o == 0:
                strict_input = (strict_input + "\n\nIMPORTANT: Your previous response did not provide a classification based on the following categories: bad, poor, regular, good, excellent. This time, ensure it aligns with the specified criteria.")
            strict_probs = extract_probs_information(answer_generation(strict_input, role_description))
            strict_first_token = strict_probs[0][0]['token'].lower()
        else:
            break
    # Security step
    if strict_first_token not in ("excellent", "good", "regular", "poor", "bad"):
        strict_first_token = original_first_token

    return strict_first_token, cleaned_tokens, token_scores_normalized