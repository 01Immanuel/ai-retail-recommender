import streamlit as st
import pandas as pd
import pickle
import surprise  # Necessary for SVD model context

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Retail Advisor", layout="wide", page_icon="🛍️")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛍️ AI Smart-Retail Recommender")
st.write("Optimizing product discovery using Hybrid AI (SVD + Content-Based Filtering)")

# --- LOAD MODELS & DATA ---
@st.cache_resource
def load_assets():
    # Loading the CSV instead of Pickle for better stability
    df = pd.read_csv('amazon_data.csv.gz')
    
    # Loading the serialised AI models
    with open('svd_model.pkl', 'rb') as f:
        svd = pickle.load(f)
    with open('cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
    with open('indices.pkl', 'rb') as f:
        indices = pickle.load(f)
        
    return df, svd, cosine_sim, indices

try:
    df, svd, cosine_sim, indices = load_assets()
    st.sidebar.success("✅ AI Models Loaded")
except Exception as e:
    st.error(f"⚠️ Deployment Error: {e}")
    st.info("Check if amazon_data.csv and the .pkl files are in the same folder on GitHub.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=100)
mode = st.sidebar.radio("Recommendation Strategy", ["Similar Products", "User Personalization", "Trending Now"])

# --- STRATEGY 1: CONTENT-BASED (Similar Products) ---
if mode == "Similar Products":
    st.subheader("🔍 Discover Similar Items")
    st.info("Uses TF-IDF & Cosine Similarity to find products with similar features.")
    
    product_names = df['product_name'].unique()
    selected_product = st.selectbox("Search for a product you like:", product_names)

    if st.button("Find Matches"):
        if selected_product in indices:
            idx = indices[selected_product]
            sim_scores = list(enumerate(cosine_sim[idx]))
            # Sort by similarity, skip the first one (it's the product itself)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
            
            st.write("### Recommended for you:")
            cols = st.columns(5)
            for i, score in enumerate(sim_scores):
                item_name = df.iloc[score[0]]['product_name']
                with cols[i]:
                    st.write(f"**{item_name}**")
                    st.caption(f"Score: {round(score[1]*100, 1)}%")
        else:
            st.error("Product index not found. Please try another item.")

# --- STRATEGY 2: COLLABORATIVE FILTERING (Personalized) ---
elif mode == "User Personalization":
    st.subheader("👤 Your Personalized Dashboard")
    st.info("Uses SVD (Matrix Factorization) to predict what you'll love next.")
    
    user_id = st.text_input("Enter User ID", placeholder="e.g., AG3D6O4...")
    
    if st.button("Generate Recommendations"):
        if user_id in df['user_id'].values:
            # 1. Get all unique product IDs
            all_products = df['product_id'].unique()
            # 2. Find what user has already interacted with
            user_interacted = df[df['user_id'] == user_id]['product_id'].unique()
            # 3. Predict for products they haven't seen
            test_items = [p for p in all_products if p not in user_interacted]
            
            predictions = [svd.predict(user_id, pid) for pid in test_items]
            predictions.sort(key=lambda x: x.est, reverse=True)
            
            st.write(f"### Top Picks for User: {user_id}")
            for pred in predictions[:5]:
                # Map ID back to Name
                p_name = df[df['product_id'] == pred.iid]['product_name'].iloc[0]
                st.write(f"🌟 **{p_name}** (Predicted Rating: {round(pred.est, 2)})")
        else:
            st.warning("New User detected! We don't have your history yet.")
            st.write("Showing Trending Products instead...")
            # Fallback to trending
            trending = df.sort_values(by='rating', ascending=False).head(5)
            for name in trending['product_name']:
                st.write(f"📈 {name}")

# --- STRATEGY 3: TRENDING (Fallback Logic) ---
else:
    st.subheader("🔥 Trending Products")
    st.write("The most popular items across our store right now.")
    # Simple popularity logic
    trending_df = df.groupby('product_name').agg({'rating': 'mean', 'user_id': 'count'}).rename(columns={'user_id': 'popularity'})
    top_trending = trending_df.sort_values(by=['popularity', 'rating'], ascending=False).head(10)
    
    for i, (name, row) in enumerate(top_trending.iterrows()):
        st.write(f"{i+1}. **{name}**")
