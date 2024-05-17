import requests
import streamlit as st
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import json
from utils import create_interactive_scatterplot
import time

state = st.session_state

if "df" not in state:
    state.df = None
if "generate_button_clicked" not in state:
    state.generate_button_clicked = False
if "barchart" not in state:
    state.barchart = None
if "chart" not in state:
    state.chart = None
if "model" not in state:
    state.model = None
if "generated_topics" not in state:
    state.generated_topics = None
if "classification_result" not in state:
    state.classification_result = None
if "review_input" not in state:
    state.review_input = ""
if "download_name" not in state:
    state.download_name = ""
if "prompts" not in state:
    state.prompts = []
if "single_review_class" not in state:
    state.single_review_class = None
if "single_review_score" not in state:
    state.single_review_score = None


api_url = 'http://localhost:8000/'
st.set_page_config(
    page_title="Customer Review Topic Modelling",
    page_icon="🎈",
    layout="wide"
)
print("state.single_review_class: ", state.single_review_class)
st.title("Customer Ticket Segmentation")


df = None

ce, c1, ce, c2, c3 = st.columns([0.07, 1.9, 0.07, 1.9, 0.07])

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
if uploaded_file is None:
    sample_data = st.checkbox(
        "Try with sample data"
    )
    if sample_data:
        df = pd.read_csv("./vividseats_reviews.csv")
        state.df = df

nr_topics = st.slider(
    "How many departments do you want?",
    min_value=1,
    max_value=10,
    step=1,
    value=5
)

# def generate_callback():
#     state.generate_button_clicked = True

if df is not None:
    print("HERE")
    st.dataframe(df)
    option = st.selectbox(
        'Select the column with the data',
        df.columns.tolist())
    
    # Check if visualization has been generated
    if state.barchart is not None and state.chart is not None:
        st.plotly_chart(state.barchart)
        st.altair_chart(state.chart, use_container_width=True)

    elif st.button("Generate Topics"):
        vectorizer_model = CountVectorizer(
            ngram_range=(1,3), stop_words="english")
        docs = df[option]

        progress_bar = st.progress(0)
        progress_text = st.empty()

        progress_text.text("Segmenting your data...")
        progress_bar.progress(10)
        model = BERTopic(
            top_n_words=6,
            min_topic_size=10,
            nr_topics=nr_topics + 1,
            vectorizer_model=vectorizer_model,
            embedding_model='paraphrase-MiniLM-L3-v2',
            low_memory=False
        )
        # Fit model
        df['topic'], probabilities = model.fit_transform(docs)
        progress_text.text(f"Found {nr_topics} separate departments!")
        progress_bar.progress(30)
        topic_info = model.get_topic_info()                
        topic_json_df = topic_info.to_json(orient='split')
        
        progress_text.text(f"Generating Visualisations...")
        progress_bar.progress(55)
        # API call to generate prompts
        response = requests.post(api_url + 'gen-prompts', json={'df_json': topic_json_df})

        if response.status_code == 200:
            prompts = response.json()['data']
            st.session_state.prompts = prompts
        else:
            st.error('Failed to generate prompts with the FastAPI server')
        
        prompts_json = json.dumps(prompts)

        # API call to generate topics
        response = requests.post(api_url + 'gen-topics', json={'array_json': prompts_json})

        if response.status_code == 200:
            generated_topics = response.json()['data']
            model.set_topic_labels(generated_topics)
        else:
            st.error('Failed to generate topics with the FastAPI server')
        progress_text.text(f"Almost there...")
        progress_bar.progress(80)
        df = df[df['topic'] != -1]
        generated_topics = generated_topics[1:]
        df['Topic_Name'] = df['topic'].apply(lambda x: generated_topics[x])
                
        barchart = model.visualize_barchart(
                            n_words = 10,
                            custom_labels = True,
                            title = "Department Word Scores",
                            width = 400,
                            height = 250,
                            autoscale = True,
                        )
        

        chart = create_interactive_scatterplot(df)
        progress_text.text(f"Complete!")
        progress_bar.progress(100)

        state.barchart = barchart
        state.chart = chart
        state.df = df
        state.model = model
        state.generated_topics = generated_topics
        state.prompts = prompts
        state.generate_button_clicked = True
                     
        st.plotly_chart(barchart)
        st.altair_chart(chart, use_container_width=True)

with st.form("classification_form"):
    # Retrieve review input from session state or use empty string if not available
    review_input = st.text_input("Write your own review and test how well the model classifies it with the new departments!", state.review_input)
    classify_button = st.form_submit_button("Classify review")
    
    if classify_button:
        generated_topics = st.session_state.generated_topics
        with st.spinner('Classifying review...'):
            model = st.session_state.model
            topic, probabilities = model.transform([review_input])
            state.single_review_class = generated_topics[topic[0]]
            state.single_review_score = probabilities[0]
            state.review_input = review_input

    st.write(f"Department: {state.single_review_class}")
    st.write(f"Confidence: {state.single_review_score}")

    download_name = st.text_input("Like the model? You can download it and use it right away! Provide the name of your custom model", state.download_name)
    download_button = st.form_submit_button("Download model")
    if download_button:
        with st.spinner('Saving model...'):
            state.download_name = download_name
            state.single_review_class = state.single_review_class
            state.single_review_score = state.single_review_score
            state.review_input = state.review_input
            model = st.session_state.model
            model.save(download_name)

            pip_install_command = "pip install bertopic"

            code_snippet = f"""
# Import BertTopic
from bertopic import BERTopic

# Load the saved model
loaded_model = BERTopic.load({download_name})

# Store the review in a variable
review = "<Write your review here>"

# Use the transform method to predict the topic for your single review
topic, probabilities = loaded_model.transform([single_review])

print('Department: ', generated_topics[topic[0]]
print('Confidence: ', probabilities[0])
"""


            st.write("Here is how to you can load and use your model:")
            st.write("Terminal")
            st.code(body=pip_install_command, language='bash')
            st.write("main.py")
            st.code(body=code_snippet, language='python', line_numbers=True)
                

                 
                


               