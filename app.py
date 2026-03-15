import html
import os
import re
import tempfile
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from huggingface_hub import snapshot_download

# ── GENRES LIST ────────────────────────────────────────────────────────────────
GENRES = [
    "Action",
    "Adventure",
    "Animation & Modeling",
    "Audio Production",
    "Casual",
    "Design & Illustration",
    "Early Access",
    "Education",
    "Free To Play",
    "Game Development",
    "Indie",
    "Massively Multiplayer",
    "Photo Editing",
    "RPG",
    "Racing",
    "Simulation",
    "Software Training",
    "Sports",
    "Strategy",
    "Utilities",
    "Video Production",
    "Web Publishing"
]


def normalize_genre_token(value):
    """Normalize a genre string so concatenated values can be matched reliably."""
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


KNOWN_GENRE_TOKENS = sorted(
    ((normalize_genre_token(genre), genre) for genre in GENRES),
    key=lambda item: len(item[0]),
    reverse=True
)
KNOWN_GENRE_LOOKUP = dict(KNOWN_GENRE_TOKENS)


def split_known_genres(value):
    """Split strings like 'ActionIndieStrategy' into known Steam genres."""
    cleaned_value = str(value).strip().strip("'\"")
    normalized = normalize_genre_token(cleaned_value)

    if not normalized:
        return []

    if normalized in KNOWN_GENRE_LOOKUP:
        return [KNOWN_GENRE_LOOKUP[normalized]]

    matches = []
    position = 0

    while position < len(normalized):
        matched = False
        for token, genre in KNOWN_GENRE_TOKENS:
            if normalized.startswith(token, position):
                matches.append(genre)
                position += len(token)
                matched = True
                break

        if not matched:
            return []

    return list(dict.fromkeys(matches))


def get_short_description(description, max_length=220):
    """Return a cleaned, short description suitable for compact card UIs."""
    cleaned = html.unescape(str(description or ""))
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned or cleaned.lower() == "nan":
        return "No description available."

    if len(cleaned) <= max_length:
        return cleaned

    trimmed = cleaned[:max_length].rsplit(" ", 1)[0].strip()
    return f"{trimmed}..."

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    
    /* ✅ Correct selector — targets the actual tab button row */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }

    /* Tabs styling */
    div[data-testid="stTabs"] button { font-size: 0.95rem; font-weight: 600; }
    /* Tabs styling */
    div[data-testid="stTabs"] button { font-size: 0.95rem; font-weight: 600; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        color: white;
    }
    .metric-card h2 { margin: 0; font-size: 1.8rem; color: #e94560; }
    .metric-card p  { margin: 0; font-size: 0.85rem; color: #aaa; }
    .game-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        color: white;
    }
    .game-card h4 { margin: 0 0 0.3rem 0; color: #e94560; font-size: 1rem; }
    .game-card p  { margin: 0; font-size: 0.82rem; color: #bbb; }
    .game-description {
        margin-top: 0.6rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 0.55rem;
    }
    .game-description summary {
        cursor: pointer;
        color: #f5c451;
        font-size: 0.82rem;
        font-weight: 600;
        list-style: none;
    }
    .game-description summary::-webkit-details-marker { display: none; }
    .game-description div {
        margin-top: 0.45rem;
        font-size: 0.8rem;
        line-height: 1.5;
        color: #d6d6d6;
    }
    .sim-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: bold;
        margin-top: 0.4rem;
    }
    .sim-high   { background: #1a4a2e; color: #4caf50; border: 1px solid #4caf50; }
    .sim-medium { background: #4a3a1a; color: #ff9800; border: 1px solid #ff9800; }
    .sim-low    { background: #4a1a1a; color: #f44336; border: 1px solid #f44336; }
    .section-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 1rem; color: #e94560; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    with st.spinner("📥 Loading Steam dataset from Hugging Face (~200MB, first load only)..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_download(
                repo_id="FronkonGames/steam-games-dataset",
                repo_type="dataset",
                allow_patterns="data/*.parquet",
                local_dir=tmpdir,
                local_dir_use_symlinks=False
            )
            data_dir = os.path.join(tmpdir, "data")
            parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
            df = pd.read_parquet(os.path.join(data_dir, parquet_files[0]))
    return df


@st.cache_data(show_spinner=False)
def preprocess(df):
    dc = df.copy()

    # Map lowercase columns from dataset to uppercase (case-insensitive mapping)
    column_mapping = {
        'price': 'Price',
        'positive': 'Positive', 
        'negative': 'Negative',
        'name': 'Name',
        'genres': 'Genres',
        'tags': 'Tags',
        'categories': 'Categories',
        'short_description': 'Short Description',
        'windows': 'Windows',
        'mac': 'Mac',
        'linux': 'Linux'
    }
    
    # Rename columns if they exist (case-insensitive check)
    for old_col, new_col in column_mapping.items():
        if old_col in dc.columns and new_col not in dc.columns:
            dc = dc.rename(columns={old_col: new_col})

    # Clean up Genres column - convert list-like strings to comma-separated with consistent spacing
    if 'Genres' in dc.columns:
        import ast
        def clean_genres(genre_str):
            try:
                # Handle None and NaN values
                if genre_str is None or (isinstance(genre_str, float) and genre_str != genre_str):  # NaN check
                    return ''

                genres_list = []
                raw_genres = []

                # Handle real Python lists from parquet/object columns.
                if isinstance(genre_str, (list, tuple, set)):
                    raw_genres = [str(g).strip().strip("'\"") for g in genre_str if str(g).strip()]
                else:
                    genre_str = str(genre_str).strip()
                    if not genre_str or genre_str.lower() == 'nan':
                        return ''

                    # Try to parse as Python list if it looks like one
                    if genre_str.startswith('['):
                        try:
                            parsed = ast.literal_eval(genre_str)
                            if isinstance(parsed, (list, tuple, set)):
                                raw_genres = [str(g).strip().strip("'\"") for g in parsed if str(g).strip()]
                        except:
                            pass

                    # If no list was parsed, try splitting by comma
                    if not raw_genres:
                        raw_genres = [g.strip().strip("'\"") for g in genre_str.split(',') if g.strip()]

                for raw_genre in raw_genres:
                    matched_genres = split_known_genres(raw_genre)
                    if matched_genres:
                        genres_list.extend(matched_genres)
                    elif raw_genre:
                        genres_list.append(raw_genre)

                # Filter out empty strings and normalize
                genres_list = [g for g in genres_list if g and len(g) > 0]
                genres_list = list(dict.fromkeys(genres_list))
                return ', '.join(genres_list) if genres_list else ''
            except Exception:
                return str(genre_str).strip()

        dc['Genres'] = dc['Genres'].apply(clean_genres)

    # Numeric columns
    for col in ['Price', 'Positive', 'Negative']:
        if col in dc.columns:
            dc[col] = pd.to_numeric(dc[col], errors='coerce').fillna(0)
        else:
            dc[col] = 0

    # Rating score
    dc['Total_Reviews'] = dc['Positive'] + dc['Negative']
    dc['Rating_Score'] = np.where(
        dc['Total_Reviews'] > 0,
        dc['Positive'] / dc['Total_Reviews'] * 100, 0
    )

    # Normalize rating for sorting
    if len(dc) > 1:  # MinMaxScaler needs at least 2 samples
        scaler = MinMaxScaler()
        dc['Rating_Norm'] = scaler.fit_transform(dc[['Rating_Score']])
    else:
        dc['Rating_Norm'] = 0.5 if len(dc) == 1 else 0.0

    # Text columns
    for col in ['Genres', 'Tags', 'Categories', 'Name', 'Short Description']:
        if col in dc.columns:
            dc[col] = dc[col].fillna('').astype(str)
        else:
            dc[col] = ''
    
    # Handle "About the game" column (might be 'detailed_description', 'short_description', or other names)
    if 'About the game' not in dc.columns:
        if 'detailed_description' in dc.columns:
            dc['About the game'] = dc['detailed_description']
        elif 'Short Description' in dc.columns:
            dc['About the game'] = dc['Short Description']
        else:
            dc['About the game'] = ''
    dc['About the game'] = dc['About the game'].fillna('').astype(str)
    dc['Short Description'] = dc['Short Description'].where(
        dc['Short Description'].str.strip() != '',
        dc['About the game']
    )

    # Platform columns
    for col in ['Windows', 'Mac', 'Linux']:
        if col in dc.columns:
            dc[col] = dc[col].fillna(False).astype(bool)
        else:
            dc[col] = False

    # Combined feature for TF-IDF
    # Build features more safely to avoid NaN propagation
    dc['combined_features'] = ''
    for col in ['Name', 'Genres', 'Genres', 'Tags', 'Categories', 'About the game']:
        if col in dc.columns:
            dc['combined_features'] += dc[col].fillna('').astype(str) + ' '
    
    dc['combined_features'] = dc['combined_features'].str.strip()
    dc['combined_features'] = dc['combined_features'].replace('', 'game')

    dc = dc.reset_index(drop=True)
    return dc


@st.cache_resource(show_spinner=False)
def build_tfidf_model(_df):
    with st.spinner("🤖 Building TF-IDF similarity matrix..."):
        # Ensure all documents have content
        documents = _df['combined_features'].fillna('game').astype(str).str.strip()
        # Replace any remaining empty strings
        documents = documents.replace('', 'game').replace('nan', 'game')
        
        # Debug: check if we have any non-empty documents
        if (documents.str.len() == 0).all():
            st.error("All documents are empty. Check your input data.")
            return None, None
        
        tfidf = TfidfVectorizer(
            stop_words=None,  # Don't filter stop words
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,  # Allow words in all documents
            token_pattern=r"(?u)\b\w+\b"
        )
        tfidf_matrix = tfidf.fit_transform(documents)
    return tfidf, tfidf_matrix


# ── RECOMMENDATION FUNCTIONS ───────────────────────────────────────────────────
def get_similar_games(game_name, df, tfidf_matrix, top_n=10):
    """Content-based: TF-IDF cosine similarity"""
    matches = df[df['Name'].str.lower() == game_name.lower()]
    if matches.empty:
        # Partial match fallback
        matches = df[df['Name'].str.lower().str.contains(game_name.lower(), na=False)]
    if matches.empty:
        return None, pd.DataFrame()

    idx = matches.index[0]
    game_vec = tfidf_matrix[idx]
    sim_scores = cosine_similarity(game_vec, tfidf_matrix).flatten()
    sim_scores[idx] = 0  # exclude itself

    top_indices = sim_scores.argsort()[::-1][:top_n]
    results = df.iloc[top_indices].copy()
    results['Similarity'] = sim_scores[top_indices]
    return matches.iloc[0], results


def get_filtered_games(df, genres=None, tags=None, price_max=60,
                       platforms=None, min_rating=0, min_reviews=0, top_n=20):
    """Feature-based filtering with support for multiple genres/tags"""
    f = df.copy()
    
    # Filter by genres (ANY match)
    if genres:
        genre_mask = pd.Series([False] * len(f), index=f.index)
        for g in genres:
            genre_mask = genre_mask | (f['Genres'].str.contains(g, case=False, na=False))
        f = f[genre_mask]
    
    # Filter by tags (ANY match)
    if tags:
        tag_mask = pd.Series([False] * len(f), index=f.index)
        for t in tags:
            tag_mask = tag_mask | (f['Tags'].str.contains(t, case=False, na=False))
        f = f[tag_mask]
    
    f = f[f['Price'] <= price_max]
    
    # Apply platform filter only if we have platforms and the columns exist
    if platforms:
        existing_platforms = [p for p in platforms if p in f.columns]
        if existing_platforms:
            mask = pd.Series([False] * len(f), index=f.index)
            for p in existing_platforms:
                mask = mask | (f[p] == True)
            f = f[mask]
        # If no platform columns exist, don't filter by platform - continue with other filters
    
    if min_rating > 0:
        f = f[f['Rating_Score'] >= min_rating]
    if min_reviews > 0:
        f = f[f['Total_Reviews'] >= min_reviews]
    return f.sort_values('Rating_Score', ascending=False).head(top_n)


def get_popular_games(df, top_n=25):
    """Popularity score = normalized rating × log(reviews)"""
    d = df[df['Total_Reviews'] >= 50].copy()
    
    # Handle empty dataframe
    if len(d) == 0:
        return pd.DataFrame()
    
    d['Popularity_Score'] = d['Rating_Norm'] * np.log1p(d['Total_Reviews'])
    
    # Only scale if we have data to scale
    if len(d) > 0:
        scaler = MinMaxScaler()
        d['Popularity_Score'] = scaler.fit_transform(d[['Popularity_Score']]) * 100
    
    return d.sort_values('Popularity_Score', ascending=False).head(top_n)


# ── UI HELPERS ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_popular_genres(df, top_n=15):
    """Extract and count genres, return sorted by frequency"""
    genres_count = {}
    
    for genre_str in df['Genres'].dropna():
        if pd.isna(genre_str):
            continue
        genre_str = str(genre_str).strip()
        if not genre_str or genre_str.lower() == 'nan':
            continue
        
        # Genres are comma-separated after preprocessing
        for g in genre_str.split(','):
            g = g.strip()
            if g and len(g) > 0:
                genres_count[g] = genres_count.get(g, 0) + 1
    
    # Sort by count descending, then alphabetically
    sorted_genres = sorted(genres_count.items(), key=lambda x: (-x[1], x[0]))
    return [g[0] for g in sorted_genres[:top_n]]

@st.cache_data(show_spinner=False)
def get_popular_tags(df, top_n=15):
    """Extract and count tags, return sorted by frequency"""
    import ast
    tags_count = {}
    
    for tag_str in df['Tags'].dropna():
        tag_str = str(tag_str).strip()
        
        # Try to parse as Python list if it looks like one
        if tag_str.startswith('['):
            try:
                tags_list = ast.literal_eval(tag_str)
                if isinstance(tags_list, list):
                    for t in tags_list:
                        t = str(t).strip().strip("'\"")
                        if t and len(t) > 1:
                            tags_count[t] = tags_count.get(t, 0) + 1
                    continue
            except:
                pass
        
        # Otherwise split by comma
        for t in tag_str.split(','):
            t = t.strip().strip("'\"[]")
            if t and len(t) > 1:
                tags_count[t] = tags_count.get(t, 0) + 1
    
    # Sort by count descending, then alphabetically
    sorted_tags = sorted(tags_count.items(), key=lambda x: (-x[1], x[0]))
    return [t[0] for t in sorted_tags[:top_n]]

def game_card(game, extra_label=None, extra_value=None):
    genres  = game.get('Genres', '')[:60]
    price   = f"${float(game.get('Price', 0)):.2f}" if float(game.get('Price', 0)) > 0 else "Free"
    rating  = f"{float(game.get('Rating_Score', 0)):.1f}%"
    reviews = f"{int(game.get('Total_Reviews', 0)):,}"
    description_text = game.get('Short Description', '') or game.get('About the game', '')
    short_description = html.escape(get_short_description(description_text))

    badge_html = ""
    if extra_label == "Similarity":
        val = float(extra_value) * 100
        cls = "sim-high" if val >= 60 else "sim-medium" if val >= 30 else "sim-low"
        badge_html = f'<span class="sim-badge {cls}">🔗 {val:.1f}% similar</span>'
    elif extra_label == "Popularity":
        badge_html = f'<span class="sim-badge sim-high">🔥 Score: {float(extra_value):.2f}</span>'

    description_html = (
        f'<details class="game-description">'
        f'<summary>Description &gt;</summary>'
        f'<div>{short_description}</div>'
        f'</details>'
    )

    st.markdown(
        f'<div class="game-card">'
        f'<h4>🎮 {game.get("Name", "Unknown")}</h4>'
        f'<p>🎯 {genres} &nbsp;|&nbsp; 💰 {price} &nbsp;|&nbsp; ⭐ {rating} &nbsp;|&nbsp; 💬 {reviews} reviews</p>'
        f'{badge_html}'
        f'{description_html}'
        f'</div>',
        unsafe_allow_html=True
    )


def stat_bar(df):
    c1, c2, c3, c4, c5 = st.columns(5)
    stats = [
        ("🎮 Total Games",   f"{len(df):,}"),
        ("💰 Avg Price",     f"${df['Price'].mean():.2f}"),
        ("⭐ Avg Rating",    f"{df[df['Rating_Score']>0]['Rating_Score'].mean():.1f}%"),
        ("🆓 Free Games",    f"{len(df[df['Price']==0]):,}"),
        ("💬 Total Reviews", f"{int(df['Total_Reviews'].sum()/1e6):.1f}M"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4, c5], stats):
        col.markdown(f"""
        <div class="metric-card">
            <h2>{value}</h2>
            <p>{label}</p>
        </div>""", unsafe_allow_html=True)


# ── MAIN APP ───────────────────────────────────────────────────────────────────
def main():
    st.markdown("<center><h1>🎮 Steam Game Recommender</h1></center>", unsafe_allow_html=True)
    st.markdown("<center><p><i>ML-powered recommendations using TF-IDF & Cosine Similarity</i></p></center>", unsafe_allow_html=True)
    st.markdown("---")

    # Load & preprocess
    raw_df = load_data()
    df     = preprocess(raw_df)
    tfidf, tfidf_matrix = build_tfidf_model(df)

    # Stats bar
    stat_bar(df)
    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍  Find Similar Games",
        "🎯  Filter & Discover",
        "🏆  Popular Games"
    ])

    # ── TAB 1: Content-Based ML Recommendations ────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Content-Based Recommendations</div>', unsafe_allow_html=True)
        st.markdown("<center>Enter a game you like and the ML model will find similar games using <b>TF-IDF vectorization + cosine similarity</b> on genres, tags, and categories.</center>", unsafe_allow_html=True)
        st.markdown("")

        if tfidf_matrix is None:
            st.error("❌ TF-IDF model failed to initialize. Check the dataset or increase min_df/max_df parameters.")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                game_input = st.text_input("🎮 Enter a game name", placeholder="e.g. Counter-Strike, Minecraft, Dota 2")
            with col2:
                top_n = st.slider("Results", 5, 20, 10)

            if st.button("🔍 Find Similar Games", type="primary"):
                if not game_input.strip():
                    st.warning("Please enter a game name.")
                else:
                    source_game, results = get_similar_games(game_input, df, tfidf_matrix, top_n)
                    if results.empty:
                        st.error(f"No game found matching '{game_input}'. Try a different name.")
                    else:
                        st.success(f"Showing {len(results)} games similar to **{source_game['Name']}**")

                        # Source game info
                        with st.expander("📋 Source Game Details", expanded=True):
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Price", f"${float(source_game.get('Price',0)):.2f}" if float(source_game.get('Price',0)) > 0 else "Free")
                            c2.metric("Rating", f"{float(source_game.get('Rating_Score',0)):.1f}%")
                            c3.metric("Reviews", f"{int(source_game.get('Total_Reviews',0)):,}")
                            st.write(f"**Genres:** {source_game.get('Genres','N/A')}")

                        # Game cards
                        for _, game in results.iterrows():
                            game_card(game, "Similarity", game['Similarity'])

    # ── TAB 2: Feature-Based Filtering ────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-title">Filter & Discover Games</div>', unsafe_allow_html=True)
        st.markdown("")

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_genres = st.multiselect("🎯 Genres (select one or more)", GENRES, 
                                            default=None,
                                            help="Games matching ANY selected genre")

        with col2:
            max_price   = st.slider("💰 Max Price ($)", 0, 100, 60)
            min_rating  = st.slider("⭐ Min Rating (%)", 0, 100, 0)
            min_reviews = st.slider("💬 Min Reviews", 0, 1000, 0)

        with col3:
            st.markdown("**🖥️ Platform**")
            win = st.checkbox("Windows", value=True)
            mac = st.checkbox("Mac")
            lin = st.checkbox("Linux")
            top_n_filter = st.slider("📋 Results to show", 5, 50, 20)

        platforms = []
        if win: platforms.append('Windows')
        if mac: platforms.append('Mac')
        if lin: platforms.append('Linux')

        button_col, spinner_col, _ = st.columns([0.15, 0.1, 0.75])
        with button_col:
            apply_filters = st.button("🎯 Apply Filters", type="primary")

        if apply_filters:
            with spinner_col:
                with st.spinner("Filtering..."):
                    results = get_filtered_games(
                        df,
                        genres=selected_genres if selected_genres else None,
                        price_max=max_price,
                        platforms=platforms,
                        min_rating=min_rating,
                        min_reviews=min_reviews,
                        top_n=top_n_filter
                    )

            if results.empty:
                st.warning("❌ No games found. Debugging info:")
                # Show what's blocking the results
                with st.expander("🔍 Why no results?"):
                    temp = df.copy()
                    st.write(f"Starting with: {len(temp)} games")
                    
                    if selected_genres:
                        genre_mask = pd.Series([False] * len(temp), index=temp.index)
                        for g in selected_genres:
                            genre_mask = genre_mask | (temp['Genres'].str.contains(g, case=False, na=False))
                        temp = temp[genre_mask]
                        st.write(f"After genre filter ({', '.join(selected_genres)}): {len(temp)} games")
                    
                    temp = temp[temp['Price'] <= max_price]
                    st.write(f"After price filter (<= ${max_price}): {len(temp)} games")
                    
                    if min_rating > 0:
                        temp = temp[temp['Rating_Score'] >= min_rating]
                        st.write(f"After rating filter (>= {min_rating}%): {len(temp)} games")
                    
                    if min_reviews > 0:
                        temp = temp[temp['Total_Reviews'] >= min_reviews]
                        st.write(f"After reviews filter (>= {min_reviews} reviews): {len(temp)} games")
            else:
                st.success(f"Nice, I found **{len(results)} games** that fit.")

                for _, game in results.iterrows():
                    game_card(game)

    # ── TAB 3: Popular Games Leaderboard ──────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-title">Popular Games Leaderboard</div>', unsafe_allow_html=True)
        st.markdown("<center>Ranked by a <b>Popularity Score</b> = Normalized Rating × log(Review Count). Rewards highly-rated games with significant community validation.</center>", unsafe_allow_html=True)
        st.markdown("")
        st.markdown("<center>🔢 Enter the number or slide to select how many top games to display:</center>", unsafe_allow_html=True)


        # Initialize session state for dynamic sync
        if 'top_n_games' not in st.session_state:
            st.session_state.top_n_games = 25

        # Centered number input (small)
        col_left, col_center, col_right = st.columns([1, 0.6, 1])
        with col_center:
            st.write("")
            top_n_input = st.number_input("Games to display", min_value=10, max_value=50, value=st.session_state.top_n_games, label_visibility="collapsed")
            st.session_state.top_n_games = top_n_input

        # Full-width slider with marks
        st.write("")
        top_n_slider = st.slider("Select number", 10, 50, int(st.session_state.top_n_games), step=1, label_visibility="collapsed")
        st.session_state.top_n_games = top_n_slider

        # Centered button (small)
        st.write("")
        col_left, col_center, col_right = st.columns([1, 0.6, 1])
        with col_center:
            show_results = st.button("📊 Show Results", type="primary", use_container_width=True)

        if show_results:
            # Use the synced value from session state
            top_n_pop = int(st.session_state.top_n_games)
            spinner_left, spinner_center, spinner_right = st.columns([1, 0.6, 1])
            with spinner_center:
                with st.spinner("Loading popular games..."):
                    popular = get_popular_games(df, top_n_pop)

            if len(popular) == 0:
                st.warning("⚠️ Not enough data for Popular Games (need games with 50+ reviews)")
            else:
                # Leaderboard cards
                for rank, (_, game) in enumerate(popular.iterrows(), 1):
                    col1, col2 = st.columns([0.08, 0.92])
                    with col1:
                        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"**#{rank}**"
                        st.markdown(f"<div style='text-align:center;padding-top:12px;font-size:1.2rem'>{medal}</div>", unsafe_allow_html=True)
                    with col2:
                        game_card(game, "Popularity", game['Popularity_Score'])


if __name__ == "__main__":
    main()
