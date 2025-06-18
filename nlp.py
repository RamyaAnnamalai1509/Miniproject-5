import streamlit as st
import pickle
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Load the trained model and vectorizer
try:
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'xgb_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    st.stop()

# Load NLTK resources if not already downloaded
try:
    stopwords.words('english')
    word_tokenize("test")
    WordNetLemmatizer()
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning and preprocessing function (same as used for training)
def clean_text(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(' +', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return ' '.join(lemmatized)

# Function to predict sentiment
def predict_sentiment(review_text):
    # Clean and preprocess the input text
    cleaned_text = clean_text(review_text)
    lemmatized_text = tokenize_and_lemmatize(cleaned_text)

    # Transform the text using the fitted TF-IDF vectorizer
    text_vector = tfidf_vectorizer.transform([lemmatized_text])

    # Predict the sentiment using the loaded model
    prediction_encoded = model.predict(text_vector)[0]

    # Map the encoded prediction back to sentiment label
    # Assuming the label encoding was 0: Negative, 1: Neutral, 2: Positive
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_sentiment = sentiment_map.get(prediction_encoded, "Unknown")

    return predicted_sentiment
def show_colored_sentiment(label):
    colors = {
        'Negative 😞': 'red',
        'Neutral 😐': 'orange',
        'Positive 😊': 'green'
    }
    color = colors.get(label, 'black')
    st.markdown(f"Predicted Sentiment: {label}", unsafe_allow_html=True)

# Sidebar menu
st.sidebar.title("Menu")
section = st.sidebar.radio("Choose Section", ["💬 Sentiment Analyzer", "📈 Data Visualizations"])

if section == "💬 Sentiment Analyzer":
    st.header("🧪 ChatGPT Feedback Analysis")
    user_input = st.text_area("✍️Enter a review about ChatGPT and instantly discover whether it’s Positive, Neutral, or Negative!")

    if st.button("Analyze"):
        if user_input.strip() == "":
            st.warning("Please enter a review.")
        else:
            sentiment = predict_sentiment(user_input)
            show_colored_sentiment(sentiment)

elif section == "📈 Data Visualizations":
    st.header("📈 ChatGPT Review Sentiment Dashboard")

    csv_file_path = "cleaned_reviews.csv"
    try:
        df = pd.read_csv(csv_file_path)
        df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    question_options = {
        "Q1. Overall Sentiment of User Reviews": "sentiment_distribution",
        "Q2. Sentiment Variation by Rating": "sentiment_by_rating",
        "Q3. Keywords by Sentiment": "sentiment_keywords",
        "Q4. Sentiment Trend Over Time (Monthly)": "sentiment_over_time",
        "Q5. Sentiment by Verified Purchase": "verified_sentiment",
        "Q6. Review Length vs Rating": "review_length_vs_rating",
        "Q7. Which locations show the most Positive or Negative Sentiment?": "location_sentiment",
        "Q8. Sentiment by Platform": "platform_sentiment",
        "Q9. Sentiment by ChatGPT Version": "version_sentiment",
        "Q10. Common Negative Feedback Themes": "negative_feedback_themes"
    }

    selected_question = st.selectbox("Select a question to visualize", list(question_options.keys()))
    selected_viz = question_options[selected_question]

    if selected_viz == "sentiment_distribution":
      st.subheader("Q1. Overall Sentiment of User Reviews")

      sentiment_counts = df['sentiment'].value_counts()

      # Show count table only
      st.table(pd.DataFrame({'Count': sentiment_counts}))

      # Bar chart with Plotly
      fig_bar = px.bar(
          x=sentiment_counts.index,
          y=sentiment_counts.values,
          labels={'x': 'Sentiment', 'y': 'Count'},
          title='Sentiment Counts Bar Chart',
          color=sentiment_counts.index,
          color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'red'})
      st.plotly_chart(fig_bar)

      # Pie chart with Plotly
      fig_pie = px.pie(
          values=sentiment_counts.values,
          names=sentiment_counts.index,
          title='Sentiment Distribution Pie Chart',
          color=sentiment_counts.index,
          color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'red'})
      st.plotly_chart(fig_pie)


    elif selected_viz == "sentiment_by_rating":
      st.subheader("Q2. Sentiment Variation by Rating")

      # Sentiment distribution by rating
      sentiment_rating_counts = df.groupby(['rating', 'sentiment']).size().reset_index(name='count')

      fig = px.bar(
          sentiment_rating_counts,
          x='rating',
          y='count',
          color='sentiment',
          title='Sentiment Distribution by Rating',
          labels={'count': 'Number of Reviews', 'rating': 'Rating', 'sentiment': 'Sentiment'},
          color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'red'},
          barmode='stack')
      st.plotly_chart(fig)

      # Define expected sentiment by rating
      def is_mismatch(row):
        if row['rating'] == 5 and row['sentiment'] != 'Positive':
          return True
        elif row['rating'] == 4 and row['sentiment'] not in ['Positive', 'Neutral']:
          return True
        elif row['rating'] == 3 and row['sentiment'] not in ['Positive', 'Neutral']:
          return True
        elif row['rating'] == 2 and row['sentiment'] not in ['Negative', 'Neutral']:
          return True
        elif row['rating'] == 1 and row['sentiment'] != 'Negative':
          return True
        return False


      # Apply mismatch filter
      mismatch = df[df.apply(is_mismatch, axis=1)]

      st.write(f"Number of mismatched reviews (rating vs sentiment): {len(mismatch)}")

      if not mismatch.empty:
        st.dataframe(mismatch[['rating', 'sentiment', 'review']].head(10))


    elif selected_viz == "sentiment_keywords":
      st.subheader("Q3. Keywords by Sentiment")

      # Word Cloud for each sentiment
      sentiments = ['Positive', 'Neutral', 'Negative']
      for sent in sentiments:
        st.markdown(f"### {sent} Reviews")
        text = " ".join(df[df['sentiment'] == sent]['lemmatized_review'])

        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

      # Function to get top words
      def get_top_words(sentiment, n=10):
        text = " ".join(df[df['sentiment'] == sentiment]['lemmatized_review'])
        tokens = text.split()
        most_common = Counter(tokens).most_common(n)
        return pd.DataFrame(most_common, columns=['Word', 'Frequency'])

      # Display top 10 words as table
      st.subheader("Top 10 Words by Sentiment")
      for sent in sentiments:
        st.markdown(f"### {sent} Sentiment")
        st.table(get_top_words(sent))


    elif selected_viz == "sentiment_over_time":
        st.subheader("Q4. Sentiment Trend Over Time (Monthly)")

        # Ensure 'date' is datetime and extract 'month'
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M').astype(str)

        # Group by month and sentiment
        monthly_sentiment = df.groupby(['month', 'sentiment']).size().reset_index(name='count')

        # Plot
        fig = px.line(monthly_sentiment,
                      x='month',
                      y='count',
                      color='sentiment',
                      title='Sentiment Trend Over Time',
                      markers=True,
                      color_discrete_map={
                          'Positive': 'green',
                          'Neutral': 'orange',
                          'Negative': 'red'})

        fig.update_layout(xaxis_title='Month', yaxis_title='Number of Reviews', hovermode='x unified')
        st.plotly_chart(fig)


    elif selected_viz == "verified_sentiment":
        st.subheader("Q5. Sentiment by Verified Purchase")
        # Group and count reviews by verified_purchase and sentiment
        verified_sentiment = df.groupby(['verified_purchase', 'sentiment']).size().reset_index(name='count')

        # Pivot table for display
        pivot_df = verified_sentiment.pivot(index='verified_purchase', columns='sentiment', values='count').fillna(0).astype(int)

        # Calculate difference row (Yes - No)
        diff_row = pivot_df.loc['Yes'] - pivot_df.loc['No']
        diff_row.name = 'Difference (Yes - No)'

        # Append difference row to pivot_df
        pivot_with_diff = pd.concat([pivot_df, diff_row.to_frame().T])

        # Bar chart using Plotly
        fig = px.bar(
            verified_sentiment,
            x='verified_purchase',
            y='count',
            color='sentiment',
            barmode='group',
            title='Sentiment Distribution by Verified Purchase',
            color_discrete_map={
                'Positive': 'green',
                'Neutral': 'orange',
                'Negative': 'red'},
            labels={'verified_purchase': 'Verified Purchase','count': 'Number of Reviews'})

        st.plotly_chart(fig)

        # Optional: Display table
        st.markdown("#### Sentiment Count Table by Verified Purchase")
        st.table(pivot_with_diff)


    elif selected_viz == "review_length_vs_rating":
        st.subheader("Q6. Review Length vs Rating")

        # Group by sentiment and calculate average review length
        avg_length = df.groupby('sentiment')['review_length'].mean().reset_index()

        # Color mapping
        color_map = {'Positive': 'green', 'Neutral': 'orange', 'Negative': 'red'}
        colors = [color_map[sent] for sent in avg_length['sentiment']]

        # Plot
        fig = px.bar(avg_length,
                     x='sentiment',
                     y='review_length',
                     color='sentiment',
                     color_discrete_map=color_map,
                     labels={'review_length': 'Average Review Length'},
                     title='Average Review Length by Sentiment')

        st.plotly_chart(fig)


    elif selected_viz == "location_sentiment":
        st.subheader("Q7. Which locations show the most Positive or Negative Sentiment?")
        # Group by location and sentiment to count reviews
        loc_sentiment = df.groupby(['location', 'sentiment']).size().reset_index(name='count')

        # Sort to show top contributing locations (optional)
        top_locations = df['location'].value_counts().head(10).index.tolist()
        loc_sentiment = loc_sentiment[loc_sentiment['location'].isin(top_locations)]

        # Plot
        fig = px.bar(loc_sentiment,
                     x='location',
                     y='count',
                     color='sentiment',
                     color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'red'},
                     title='Sentiment by Location',
                     labels={'count': 'Review Count', 'location': 'Location'},
                     barmode='group')

        st.plotly_chart(fig)

        sentiment_table = df[df['location'].isin(top_locations)].pivot_table(
            index='location',
            columns='sentiment',
            values='review',
            aggfunc='count',
            fill_value=0).reset_index()

        # Reorder columns for clarity
        sentiment_table = sentiment_table[['location', 'Positive', 'Neutral', 'Negative']]

        # Show the table
        st.dataframe(sentiment_table)


    elif selected_viz == "platform_sentiment":
        st.subheader("Q8. Sentiment by Platform")
        # Count of sentiments per platform
        platform_sentiment = df.groupby(['platform', 'sentiment']).size().reset_index(name='count')
        pivot_platform = platform_sentiment.pivot(index='sentiment', columns='platform', values='count').fillna(0).astype(int)

        # Add a row with difference (Positive - Negative) to understand imbalance
        pivot_platform['Difference (Mobile - Web)'] = pivot_platform.get('Mobile', 0) - pivot_platform.get('Web', 0)

        # Show table
        st.dataframe(pivot_platform.reset_index())

        # Plot bar chart using Plotly
        fig = px.bar(platform_sentiment,
                     x='platform',
                     y='count',
                     color='sentiment',
                     barmode='group',
                     color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'red'},
                     title="Sentiment Distribution by Platform")

        st.plotly_chart(fig)


    elif selected_viz == "version_sentiment":
        st.subheader("Q9. Sentiment by ChatGPT Version")
        # Group by version and sentiment
        version_sentiment = df.groupby(['version', 'sentiment']).size().reset_index(name='count')

        # Pivot for table view
        pivot_version = version_sentiment.pivot(index='version', columns='sentiment', values='count').fillna(0).astype(int)

        # Add total reviews and most frequent sentiment per version
        pivot_version['Total Reviews'] = pivot_version.sum(axis=1)
        pivot_version['Top Sentiment'] = pivot_version[['Positive', 'Neutral', 'Negative']].idxmax(axis=1)

        st.dataframe(pivot_version.reset_index())

        # Plotly stacked bar chart
        fig = px.bar(version_sentiment,
                     x='version',
                     y='count',
                     color='sentiment',
                     title='Sentiment Distribution by Version',
                     color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'red'})
        st.plotly_chart(fig)


    elif selected_viz == "negative_feedback_themes":
        st.subheader("Q10. Common Negative Feedback Themes")
        # Filter negative reviews
        negative_reviews = df[df['sentiment'] == 'Negative']['lemmatized_review']

        # Join all reviews and split into tokens
        negative_text = " ".join(negative_reviews).split()

        # Get most common keywords
        top_negative_words = Counter(negative_text).most_common(20)
        top_negative_df = pd.DataFrame(top_negative_words, columns=['Keyword', 'Frequency'])

        st.markdown("### Top 20 Keywords in Negative Reviews")
        st.table(top_negative_df)

        # WordCloud for visualization
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(" ".join(negative_text))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

     