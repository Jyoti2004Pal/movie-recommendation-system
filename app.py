import streamlit as st
from movie_recommender import MovieRecommender

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f0f1a;
        color: #f0f0f0;
    }

    /* Title */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #e50914, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        text-align: center;
        color: #aaaaaa;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Movie card */
    .movie-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e5091433;
        border-left: 4px solid #e50914;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .movie-title-text {
        font-size: 1.05rem;
        font-weight: 600;
        color: #ffffff;
    }

    .movie-rank {
        font-size: 1.4rem;
        font-weight: 800;
        color: #e50914;
        min-width: 36px;
        margin-right: 1rem;
    }

    .score-badge {
        background: #e5091422;
        border: 1px solid #e50914;
        color: #ff6b6b;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        white-space: nowrap;
    }

    /* Search box label */
    label {
        color: #cccccc !important;
        font-size: 0.95rem !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #e50914, #ff4444);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85;
    }

    /* Divider */
    hr {
        border-color: #333355;
        margin: 1.5rem 0;
    }

    /* Info box */
    .info-box {
        background: #1a1a2e;
        border: 1px solid #333355;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #aaaaaa;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached so it only runs once) ───────────────────────────────────
@st.cache_resource
def load_model():
    recommender = MovieRecommender()
    recommender.load_data("data.csv")
    recommender.build_similarity_matrix()
    return recommender


# ── App layout ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 Movie Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Discover movies similar to your favourites using AI</div>', unsafe_allow_html=True)

# Load model with a spinner
with st.spinner("⏳ Loading movie database... (first load takes ~10 seconds)"):
    recommender = load_model()

all_movies = recommender.get_all_movies()

st.markdown("---")

# ── Input section ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    selected_movie = st.selectbox(
        "🎥 Pick a movie you love:",
        options=sorted(all_movies),
        index=None,
        placeholder="Start typing a movie name..."
    )

with col2:
    num_recs = st.selectbox(
        "How many?",
        options=[5, 8, 10],
        index=0
    )

get_recs = st.button("🍿 Get Recommendations")

# ── Results section ────────────────────────────────────────────────────────────
if get_recs:
    if not selected_movie:
        st.warning("⚠️ Please select a movie first!")
    else:
        with st.spinner(f"Finding movies similar to **{selected_movie}**..."):
            results = recommender.recommend(selected_movie, num_recommendations=num_recs)

        if results and "error" not in results[0]:
            st.markdown(f"### 🎯 Top {len(results)} movies similar to *{selected_movie}*")
            st.markdown("---")

            for i, rec in enumerate(results, start=1):
                score_percent = int(rec["similarity_score"] * 100)
                st.markdown(f"""
                <div class="movie-card">
                    <div style="display:flex; align-items:center;">
                        <span class="movie-rank">#{i}</span>
                        <span class="movie-title-text">{rec['movie']}</span>
                    </div>
                    <span class="score-badge">Match: {score_percent}%</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                💡 Recommendations are based on genres, keywords, cast, director, and overview similarity.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error(f"❌ Movie not found. Try a different title.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#555577; font-size:0.8rem;">'
    'Built with ❤️ using Python · Scikit-learn · Streamlit'
    '</div>',
    unsafe_allow_html=True
)
