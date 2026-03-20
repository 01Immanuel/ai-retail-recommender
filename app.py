import streamlit as st
import pandas as pd
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Retail Advisor", layout="wide")
st.title("🛍️ AI Product Recommender")
st.write("This system suggests products using Collaborative Filtering and Content-Based models.")

# --- LOAD MODELS & DATA ---
@st.cache_resource
def load_assets():
    # These files must be in the same folder as this script
    df = pd.read_pickle('amazon_data.pkl')
    svd = pickle.load(open('svd_model.pkl', 'rb'))
    cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))
    indices = pickle.load(open('indices.pkl', 'rb'))
    return df, svd, cosine_sim, indices

# Initialize assets
try:
    df, svd, cosine_sim, indices = load_assets()
except FileNotFoundError:
    st.error("Deployment Files (.pkl) not found! Please ensure you have completed Step A.")
    st.stop()

# --- SIDEBAR CONTROL ---
st.sidebar.header("Recommendation Strategy")
mode = st.sidebar.radio("Select Mode", ["Discover by Product", "Personalized for User"])

# --- STRATEGY 1: CONTENT-BASED (Product Similarity) ---
if mode == "Discover by Product":
    st.subheader("Find Similar Items")
    # Filters based on your 'main_category' feature 
    category = st.selectbox("Filter by Category", df['main_category'].unique())
    names = df[df['main_category'] == category]['product_name'].unique()
    selected_product = st.selectbox("Select a product you like:", names)

    if st.button("Recommend Similar"):
        # Get index for Content-Based filtering [cite: 8]
        idx = indices[selected_product]
        
        # Calculate similarity scores using the pre-computed matrix [cite: 7]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        
        # Extract top 5 recommendations [cite: 9]
        product_indices = [i[0] for i in sim_scores]
        recommendations = df['product_name'].iloc[product_indices].tolist()
        
        st.write("### Because you liked that, you might also like:")
        for rec in recommendations:
            st.write(f"✅ {rec}")

# --- STRATEGY 2: COLLABORATIVE FILTERING (User-Based) ---
else:
    st.subheader("Personalized Recommendations")
    user_id = st.text_input("Enter User ID", placeholder="e.g., AG3D6O4...")

    if st.button("Get My Picks"):
        if user_id in df['user_id'].values:
            st.success(f"Predicting top picks for {user_id}...")
            
            # Identify unrated items for this user [cite: 16]
            all_products = df['product_id'].unique()
            user_interacted = df[df['user_id'] == user_id]['product_id'].unique()
            unrated = [p for p in all_products if p not in user_interacted]
            
            # Predict ratings using the SVD model [cite: 15]
            predictions = [svd.predict(user_id, pid) for pid in unrated]
            predictions.sort(key=lambda x: x.est, reverse=True)
            
            # Display top 5 personalized results [cite: 16]
            top_5 = predictions[:5]
            st.write("### Your Top Personalized Picks:")
            for p in top_5:
                p_name = df[df['product_id'] == p.iid]['product_name'].iloc[0]
                st.write(f"🌟 {p_name}")
        else:
            st.warning("User ID not found in the system.")