import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import ast
from typing import List, Dict, Tuple

class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.similarity_matrix = None
        self.indices = None

    def load_data(self, filepath: str):
        """Load and preprocess the movie dataset"""
        print("Loading dataset...")
        file_id = "1MJw_O3S_BEL7rWGNujilB86S0E1qNvaR"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        self.movies_df = pd.read_csv(url)
        

        # Select relevant columns
        required_cols = ['title', 'genres', 'keywords', 'cast', 'crew', 'overview']
        for col in required_cols:
            if col not in self.movies_df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")

        # Fill NaN values
        self.movies_df = self.movies_df[required_cols].fillna('')

        # Extract director from crew
        self.movies_df['director'] = self.movies_df['crew'].apply(self._extract_director)

        # Extract top 3 cast members
        self.movies_df['cast'] = self.movies_df['cast'].apply(self._extract_cast)

        # Extract top 5 genres
        self.movies_df['genres'] = self.movies_df['genres'].apply(self._extract_top_items)

        # Extract top 5 keywords
        self.movies_df['keywords'] = self.movies_df['keywords'].apply(self._extract_top_items)

        # Create a combined feature string
        self.movies_df['combined_features'] = (
            self.movies_df['genres'].astype(str) + ' ' +
            self.movies_df['keywords'].astype(str) + ' ' +
            self.movies_df['cast'].astype(str) + ' ' +
            self.movies_df['director'].astype(str) + ' ' +
            self.movies_df['overview'].astype(str)
        )

        # Create similarity index mapping
        self.indices = pd.Series(
            self.movies_df.index,
            index=self.movies_df['title'].str.lower().str.strip()
        ).drop_duplicates()

        print(f"Loaded {len(self.movies_df)} movies")
        return self.movies_df

    def _extract_director(self, crew_str: str) -> str:
        """Extract director name from crew JSON"""
        try:
            crew_list = ast.literal_eval(crew_str) if isinstance(crew_str, str) else []
            for member in crew_list:
                if member.get('job') == 'Director':
                    return member.get('name', '')
        except:
            pass
        return ''

    def _extract_cast(self, cast_str: str, top_n: int = 3) -> str:
        """Extract top N cast members"""
        try:
            cast_list = ast.literal_eval(cast_str) if isinstance(cast_str, str) else []
            names = [member.get('name', '') for member in cast_list[:top_n]]
            return ' '.join(names)
        except:
            return ''

    def _extract_top_items(self, items_str: str, top_n: int = 5) -> str:
        """Extract top N items from genres/keywords"""
        try:
            items_list = ast.literal_eval(items_str) if isinstance(items_str, str) else []
            names = [item.get('name', '') for item in items_list[:top_n]]
            return ' '.join(names)
        except:
            return ''

    def build_similarity_matrix(self):
        """Build cosine similarity matrix from combined features"""
        print("Building similarity matrix...")

        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            lowercase=True
        )

        tfidf_matrix = tfidf.fit_transform(self.movies_df['combined_features'])
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        print(f"Similarity matrix built: {self.similarity_matrix.shape}")
        return self.similarity_matrix

    def recommend(self, movie_title: str, num_recommendations: int = 10) -> List[Dict[str, any]]:
        """
        Get movie recommendations based on given movie title

        Args:
            movie_title: Name of the movie
            num_recommendations: Number of recommendations to return

        Returns:
            List of dictionaries with movie title and similarity score
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not built. Call build_similarity_matrix() first.")

        title_lower = movie_title.strip().lower()

        if title_lower not in self.indices:
            # Try to find closest match
            similar_titles = [
                t for t in self.indices.index
                if title_lower in t or t in title_lower
            ]
            if similar_titles:
                title_lower = similar_titles[0]
            else:
                return [{"error": f"Movie '{movie_title}' not found in database"}]

        idx = self.indices[title_lower]

        # Get similarity scores for this movie
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity score (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top N (excluding the movie itself)
        sim_scores = [s for s in sim_scores if s[0] != idx][:num_recommendations]

        # Get movie indices and scores
        movie_indices = [i[0] for i in sim_scores]
        similarity_scores = [float(i[1]) for i in sim_scores]

        # Get movie titles
        recommendations = self.movies_df.iloc[movie_indices]['title'].tolist()

        # Format response
        result = []
        for title, score in zip(recommendations, similarity_scores):
            result.append({
                "movie": title,
                "similarity_score": round(score, 4)
            })

        return result

    def get_all_movies(self) -> List[str]:
        """Get list of all movie titles in database"""
        if self.movies_df is not None:
            return self.movies_df['title'].tolist()
        return []

# Quick test function
def quick_test():
    """Test the recommender with sample input"""
    recommender = MovieRecommender()
    # You'll need to provide the path to TMDB dataset
    # recommender.load_data('data/tmdb_5000_movies.csv')
    # recommender.build_similarity_matrix()
    # results = recommender.recommend('Avengers')
    # print(results)
    pass

if __name__ == "__main__":
    quick_test()
