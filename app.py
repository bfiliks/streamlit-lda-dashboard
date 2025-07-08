import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from gensim import corpora, models
from gensim.utils import simple_preprocess
from sklearn.manifold import TSNE
from textblob import TextBlob
import re

# --- Load and preprocess comments ---
st.title("UNDERSIEGE YouTube Comments Dashboard")
st.markdown("Visualize topics, sentiment, and download insights from viewer responses")

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

    topic_array = np.array(topic_vectors)

    # --- t-SNE Dimensionality Reduction ---
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(topic_array)

    # Assign dominant topic
    dominant_topic = [np.argmax(tv) for tv in topic_vectors]

    # Add sentiment analysis
    def get_sentiment(text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"

    sentiments = comments.apply(get_sentiment)

    # Map topic numbers to labels
    topic_labels = {
        0: "Worship & Music",
        1: "Spiritual Warfare",
        2: "Encouragement",
        3: "Repentance & End Times",
        4: "Testimonies & Gratitude"
    }

    # Build DataFrame for visualization
    tsne_df = pd.DataFrame({
        'Comment': comments.values,
        'Topic': dominant_topic,
        'tSNE-1': tsne_results[:, 0],
        'tSNE-2': tsne_results[:, 1],
        'Sentiment': sentiments
    })
    tsne_df["Topic_Label"] = tsne_df["Topic"].map(topic_labels)

    # --- Sidebar Filters ---
    st.sidebar.header("🔧 Filter Options")
    selected_topic = st.sidebar.multiselect("Filter by Topic Label", options=tsne_df['Topic_Label'].unique(), default=tsne_df['Topic_Label'].unique())
    selected_sentiment = st.sidebar.multiselect("Filter by Sentiment", options=tsne_df['Sentiment'].unique(), default=tsne_df['Sentiment'].unique())

    filtered_df = tsne_df[(tsne_df['Topic_Label'].isin(selected_topic)) & (tsne_df['Sentiment'].isin(selected_sentiment))]

    # --- Plotly Interactive Scatter ---
    color_option = st.selectbox("Color plot by", ["Topic_Label", "Sentiment"])

    fig = px.scatter(
        filtered_df,
        x='tSNE-1',
        y='tSNE-2',
        color=color_option,
        hover_data=['Comment', 'Topic_Label', 'Sentiment'],
        title=f't-SNE Visualization of Comments by {color_option}'
    )
    st.plotly_chart(fig)

    # --- Keyword Search ---
    st.subheader("🔍 Keyword Search in Comments")
    search_input = st.text_input("Enter keyword(s) separated by commas (e.g. Jesus, Daniel, music)", "")

    keyword_df = filtered_df.copy()

    if search_input:
        # Process multiple keywords
        keywords = [kw.strip() for kw in search_input.split(",") if kw.strip()]
        pattern = '|'.join(re.escape(kw) for kw in keywords)

        # Filter by keyword presence (case-insensitive)
        mask = keyword_df['Comment'].str.contains(pattern, case=False, na=False)
        keyword_df = keyword_df[mask]

        st.success(f"Found {len(keyword_df)} comment(s) containing: {', '.join(keywords)}")

        # Highlight using regex (preserving original casing)
        def highlight_terms(comment):
            def replacer(match):
                return f"**:orange[`{match.group(0)}`]**"
            return re.sub(pattern, replacer, comment, flags=re.IGNORECASE)

        st.markdown("### Matched Comments (Highlighted)")
        for i, (_, row) in enumerate(keyword_df.iterrows()):
            if i >= 50:
                st.warning("Too many results! Showing only first 50 matches.")
                break
            st.markdown(f"- {highlight_terms(row['Comment'])}")
    else:
        st.info("Showing all comments (no keyword filter applied)")

    # --- Filtered Table Display ---
    st.subheader("Explore Comments Table")
    st.dataframe(keyword_df[['Comment', 'Topic_Label', 'Sentiment']])

    # --- Download button ---
    st.download_button(
        label="📥 Download Filtered Comments as CSV",
        data=keyword_df[['Comment', 'Topic_Label', 'Sentiment']].to_csv(index=False),
        file_name="filtered_comments.csv",
        mime="text/csv"
    )
