# 🎮 Steam Game Recommender

An ML-powered game recommendation system built with Python and Streamlit, analyzing **80,000+ Steam games**.

## 🔗 Live Demo
[Click here to try the app](YOUR_STREAMLIT_URL_HERE)

## 🤖 ML Techniques Used
- **TF-IDF Vectorization** — converts game genres, tags, and categories into numerical feature vectors
- **Cosine Similarity** — measures similarity between games in vector space to find the most relevant recommendations
- **MinMaxScaler** — normalizes rating scores for fair popularity ranking
- **Popularity Scoring** — custom formula combining normalized ratings and log-scaled review counts

## ✨ Features
- **Find Similar Games** — type any game name and get ML-powered recommendations with similarity scores
- **Filter & Discover** — filter by genre, tags, price, rating, platform
- **Popular Games Leaderboard** — ranked by a composite popularity score with genre breakdown charts

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (TF-IDF, Cosine Similarity, MinMaxScaler)
- Streamlit
- Plotly
- Hugging Face Hub (dataset loading)

## 📦 Dataset
[Steam Games Dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset) — loaded at runtime from Hugging Face (~200MB, cached after first load).

## 🚀 Run Locally
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/steam-game-recommender
cd steam-game-recommender

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 👤 Author
Rishabh Rajput — [LinkedIn](https://linkedin.com/in/rishabhrajput15)
