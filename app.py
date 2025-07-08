import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from gensim import corpora, models
from gensim.utils import simple_preprocess
from sklearn.manifold import TSNE

# --- Load and preprocess comments ---
st.title("UNDERSIEGE YouTube Comments Dashboard")
st.markdown("Visualize and explore themes from viewer responses")

# Upload CSV
uploaded_file = st.file_uploader("Upload your YouTube comments CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    comments = df['text'].dropna().astype(str)

    # Preprocess comments
    basic_stopwords = {"the", "and", "is", "in", "to", "of", "for", "a", "on", "this", "that", "it", "with"}
    processed_comments = comments.apply(
        lambda x: [word for word in simple_preprocess(x) if word not in basic_stopwords]
    )

    # --- LDA Topic Modeling ---
    dictionary = corpora.Dictionary(processed_comments)
    corpus = [dictionary.doc2bow(text) for text in processed_comments]

    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=5,
        passes=10,
        random_state=42
    )

    # Get topic vectors
    topic_vectors = []
    for doc_bow in corpus:
        topic_dist = lda_model.get_document_topics(doc_bow, minimum_probability=0)
        topic_vec = [prob for _, prob in topic_dist]
        topic_vectors.append(topic_vec)

    # Convert to NumPy array
    topic_array = np.array(topic_vectors)

    # --- t-SNE Dimensionality Reduction ---
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(topic_array)

    # Assign dominant topic
    dominant_topic = [np.argmax(tv) for tv in topic_vectors]

    # Build DataFrame for Plotly
    tsne_df = pd.DataFrame({
        'Comment': comments.values,
        'Topic': dominant_topic,
        'tSNE-1': tsne_results[:, 0],
        'tSNE-2': tsne_results[:, 1]
    })

    # --- Plotly Interactive Scatter ---
    fig = px.scatter(
        tsne_df,
        x='tSNE-1',
        y='tSNE-2',
        color=tsne_df['Topic'].astype(str),
        hover_data=['Comment'],
        title='t-SNE Visualization of LDA Topics'
    )
    st.plotly_chart(fig)

    # Optional: Filter and view topic-wise
    topic_choice = st.selectbox("Filter by Topic ID", sorted(tsne_df['Topic'].unique()))
    st.dataframe(tsne_df[tsne_df['Topic'] == topic_choice][['Comment', 'Topic']])
