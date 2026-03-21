
# 🛍️ AI-Powered Retail Recommendation System
[WebApp link](https://ai-retail-recommender-h84gnkgiehykjzlu3c5tqv.streamlit.app/)


## 📌 Project Overview

In the modern e-commerce landscape, "Product Overload" often leads to decision fatigue and lost sales. This Capstone Project delivers a **Hybrid Recommendation Engine** designed to increase conversion rates and enhance user discovery by providing highly relevant product suggestions.

The system analyzes user behavior and product metadata to bridge the gap between thousands of available items and the specific needs of a customer.

## 🚀 Key Features

  * **Collaborative Filtering (SVD):** Predicts user preferences based on historical interaction patterns using Singular Value Decomposition.
  * **Content-Based Filtering (TF-IDF):** Suggests products with similar features (descriptions, categories, and titles) using Cosine Similarity.
  * **Hybrid Logic:** Seamlessly switches between models to handle both existing users and "Cold Start" scenarios.
  * **Interactive Web Dashboard:** A sleek, user-friendly interface built with **Streamlit** for real-time recommendations.

## 🏗️ System Architecture

The project utilizes a multi-layered approach to ensure accuracy:

1.  **Data Layer:** Processing and cleaning of the Amazon Retail Dataset.
2.  **Modeling Layer:** \* `Scikit-Surprise` for matrix factorization.
      * `Scikit-Learn` for text vectorization and similarity matrices.
3.  **Deployment Layer:** Serialized models (.pkl) served via a Python-based web application.

## 🛠️ Tech Stack

  * **Language:** Python 3.12
  * **Libraries:** Pandas, NumPy, Scikit-learn, Scikit-surprise
  * **Web Framework:** Streamlit
  * **Deployment:** GitHub & Streamlit Community Cloud

## 📥 Installation & Local Setup

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/ai-retail-recommender.git
cd ai-retail-recommender
```

### 2\. Set Up Virtual Environment

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Run the Application

```bash
streamlit run webapp.py
```

## 📂 Project Structure

```text
├── webapp.py              # Main Streamlit application
├── requirements.txt       # Project dependencies
├── amazon_data.pkl        # Cleaned dataset (Serialized)
├── svd_model.pkl          # Trained Collaborative Filtering model
├── cosine_sim.pkl         # Content-Based Similarity matrix
├── indices.pkl            # Product name mapping
└── README.md              # Project documentation
```

## 📊 Evaluation Metrics

The models were evaluated based on:

  * **RMSE (Root Mean Square Error):** To measure the accuracy of predicted ratings.
  * **Precision @ K:** To ensure the relevance of the top 5 recommended products.

## 👥 Contributors
This project was developed as a collaborative Capstone effort by:
* **Emmanuel Akanbi** - *Lead Developer / Data Integration* - [GitHub Profile](https://github.com/01Immanuel)
* **Daniel Akanbi** - *Co-Developer / Hybrid Model Architecture* - [GitHub Profile](https://github.com/Ak-Dan)
This is a Capstone Project. Suggestions and feedback are welcome\! Please open an issue or submit a pull request for any improvements.

-----

