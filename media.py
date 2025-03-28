import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go


# Set up Dashboard
st.title("Social Media Sentiments Analysis Dashboard")
st.text("Sentiment analysis is the process of analyzing large volumes of text to determine whether it expresses a positive, negative or neutral sentiment.")
st.sidebar.title("Social Media Sentiments Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv("sentimentdataset.csv", encoding="utf-8")  # Changed to relative path
    df["Platform"] = df["Platform"].str.title()
    return df

df = load_data()

# Get unique platforms
unique_platforms = sorted(df["Platform"].unique()) 

# Find the most liked post for each platform
most_liked_by_platform = df.loc[df.groupby("Platform")["Likes"].idxmax()]

# Sidebar - Radio selection for platforms
st.sidebar.subheader("Most Liked Post by Platform")
selected_platform_radio = st.sidebar.radio(
    "Select Platform to View Top Post",
    options=unique_platforms,
    key="platform_radio"
)

# Display the most liked post for the selected platform
platform_post = most_liked_by_platform[most_liked_by_platform["Platform"] == selected_platform_radio]
if not platform_post.empty:
    post = platform_post.iloc[0]
    st.sidebar.success(
        f"**{selected_platform_radio}** (Most Liked Post):\n\n"
        f"**👍 Likes:** {post['Likes']}\n\n"
        f"**📝 Post:** {post['Text']}\n\n"
        f"**👤 User:** {post['User']}"
    )
else:
    st.sidebar.warning(f"No posts found for {selected_platform_radio}.")

# Country filter
selected_country = st.sidebar.selectbox("Filter by Country", df["Country"].dropna().unique())
filtered_df = df[df["Country"] == selected_country]

# Date filters
selected_year = st.sidebar.slider("Select Year", int(df["Year"].min()), int(df["Year"].max()), int(df["Year"].max()))
selected_month = st.sidebar.selectbox("Select Month", df["Month"].unique())
selected_day = st.sidebar.slider("Select Day", 1, 31, 1)

filtered_df = df[(df["Year"] == selected_year) & (df["Month"] == selected_month) & (df["Day"] == selected_day)]

# Sentiment analysis
st.header("Sentiment Analysis")

# Top 10 sentiments
top_10_sentiments = df["Sentiment"].value_counts().nlargest(10).index
sentiment_filtered_df = df[df["Sentiment"].isin(top_10_sentiments)]

# Sentiment distribution plot (Top 10)
st.subheader("Sentiment Distribution (Top 10)")
fig1 = px.bar(
    sentiment_filtered_df,
    x="Sentiment",
    title="Sentiment Distribution (Top 10)",
    category_orders={"Sentiment": top_10_sentiments},
    color_discrete_sequence=px.colors.sequential.Viridis
)
fig1.update_layout(xaxis_title="Sentiment", yaxis_title="Count")
st.plotly_chart(fig1)

# Posts by platform
st.subheader("Posts by Platform")
fig2 = px.bar(
    df,
    x="Platform",
    title="Posts by Platform",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig2.update_layout(xaxis_title="Platform", yaxis_title="Count")
st.plotly_chart(fig2)

# Average likes per platform
st.subheader("Average Likes per Platform")
platform_likes = df.groupby("Platform", as_index=False)["Likes"].mean()
fig3 = px.bar(
    platform_likes,
    x="Platform",
    y="Likes",
    title="Average Likes per Platform",
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig3.update_layout(yaxis_title="Average Likes")
st.plotly_chart(fig3)

# Average likes per sentiment
st.subheader("Average Likes per Sentiment (Top 10)")
sentiment_likes = sentiment_filtered_df.groupby("Sentiment", as_index=False)["Likes"].mean()
fig4 = px.bar(
    sentiment_likes,
    x="Sentiment",
    y="Likes",
    title="Average Likes per Sentiment (Top 10)",
    category_orders={"Sentiment": top_10_sentiments},
    color_discrete_sequence=px.colors.sequential.Plasma
)
fig4.update_layout(yaxis_title="Average Likes")
st.plotly_chart(fig4)

# Sentiment-platform summary table
st.subheader("Sentiment Distribution by Platform")
sentiment_platform_counts = df.pivot_table(index="Platform", columns="Sentiment", aggfunc="size", fill_value=0)
st.dataframe(sentiment_platform_counts)

# ========== ENHANCED INTERACTIVE MAP ==========
st.header("🌍 Sentiment by Country")

# Create aggregated data
country_sentiment = df.groupby(['Country', 'Sentiment']).size().unstack(fill_value=0)
country_sentiment['Total_Posts'] = country_sentiment.sum(axis=1)
country_sentiment['Dominant_Sentiment'] = country_sentiment.idxmax(axis=1)
country_sentiment = country_sentiment.reset_index()

# Create interactive choropleth map
fig_map = px.choropleth(
    country_sentiment,
    locations="Country",
    locationmode='country names',
    color="Total_Posts",
    hover_name="Country",
    hover_data={
        'Total_Posts': True,
        'Dominant_Sentiment': True,
        **{sentiment: True for sentiment in country_sentiment.columns[1:-2]}
    },
    color_continuous_scale=px.colors.sequential.Plasma,
    title="<b>Interactive Sentiment Analysis by Country</b>",
    height=600,
    template='plotly_dark'
)

# Map configuration
fig_map.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth',
        landcolor='lightgray',
        lakecolor='rgba(0,0,100,0.2)'
    ),
    margin={"r":0,"t":40,"l":0,"b":0},
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
    ),
    coloraxis_colorbar=dict(
        title="Post Count",
        thickness=20,
        len=0.75
    )
)

# Country selection dropdown
country_list = sorted(country_sentiment['Country'].unique())
selected_map_country = st.selectbox(
    "Search or select country to zoom:",
    country_list,
    index=country_list.index('United States') if 'United States' in country_list else 0
)

# Add drill-down tabs
tab1, tab2 = st.tabs(["📊 Sentiment Breakdown", "🆚 Country Comparison"])

with tab1:
    # Sentiment breakdown for selected country
    country_data = country_sentiment[country_sentiment['Country'] == selected_map_country]
    if not country_data.empty:
        melted_data = country_data.melt(
            id_vars=['Country'],
            value_vars=country_sentiment.columns[1:-2],
            var_name='Sentiment',
            value_name='Count'
        )
        
        fig_country = px.bar(
            melted_data,
            x='Sentiment',
            y='Count',
            color='Sentiment',
            title=f"Detailed Sentiment Distribution in {selected_map_country}",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text_auto=True
        )
        fig_country.update_layout(showlegend=False)
        st.plotly_chart(fig_country, use_container_width=True)
    else:
        st.warning("No data available for selected country")

with tab2:
    # Compare multiple countries
    compare_countries = st.multiselect(
        "Select up to 5 countries to compare:",
        country_list,
        default=['United States', 'United Kingdom'] if {'United States', 'United Kingdom'}.issubset(country_list) else country_list[:2],
        max_selections=5
    )
    
    if compare_countries:
        compare_data = country_sentiment[country_sentiment['Country'].isin(compare_countries)]
        # Melt the data properly for comparison
        melted_compare = compare_data.melt(
            id_vars=['Country'], 
            value_vars=compare_data.columns[1:-2],  # Exclude Total_Posts and Dominant_Sentiment
            var_name='Sentiment', 
            value_name='Count'
        )
        
        fig_compare = px.bar(
            melted_compare,
            x='Country',
            y='Count',
            color='Sentiment',  # Now this matches the melted column name
            barmode='group',
            title="Sentiment Comparison Between Countries",
            labels={'Count': 'Post Count'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.info("Please select countries to compare")

# Display the enhanced map
st.plotly_chart(fig_map, use_container_width=True)

# ========== WORD CLOUD VISUALIZATION ==========
st.header("☁️ Most Frequent Words in Posts")

# Preprocess text data for word cloud
def preprocess_text(text):
    text = str(text).lower()
    text = ' '.join([word for word in text.split() if len(word) > 3])
    return text

# Combine all text
all_text = ' '.join(df['Text'].apply(preprocess_text))

# Generate word cloud
wordcloud = WordCloud(
    width=1200, 
    height=600,
    background_color='white',
    colormap='plasma',
    max_words=300,
    stopwords=None,
    contour_width=0,
    contour_color='white'
).generate(all_text)

# Display in Streamlit
st.image(wordcloud.to_array(), use_column_width=True)