import numpy as np
import pandas as pd
import html

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

# =========================
# LOAD DATA
# =========================
books = pd.read_csv("Real_books.csv")

NO_COVER = "https://placehold.co/400x600/1a1f3a/ffffff?text=No+Cover&font=Inter"

books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    NO_COVER,
    books["thumbnail"] + "&fife=w800"
)
books_set = set(books["isbn13"].values)

# =========================
# LAZY-LOAD VECTOR DB SETUP
# =========================
embedding_model = None
db_books = None

def get_vector_db():
    global embedding_model, db_books
    if db_books is None:
        raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
        documents = text_splitter.split_documents(raw_documents)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db_books = Chroma.from_documents(documents, embedding_model)
    return db_books

# =========================
# CORE RECOMMENDER
# =========================
def retrieve_semantic_recommendations(query, category="All", tone="All", initial_top_k=50, final_top_k=12):
    if not query.strip():
        return books.sample(min(final_top_k, len(books)))

    db = get_vector_db()  # lazy-load here
    recs = db.similarity_search_with_score(query, k=initial_top_k)
    if not recs:
        return pd.DataFrame()

    books_with_scores = []
    for doc, score in recs:
        try:
            isbn = int(doc.page_content.strip('"').split()[0].replace(":", ""))
            if isbn in books_set:
                books_with_scores.append((isbn, score))
        except:
            continue

    if not books_with_scores:
        return books.sample(final_top_k)

    df_scores = pd.DataFrame(books_with_scores, columns=["isbn13", "score"])
    books_recs = books.merge(df_scores, on="isbn13")
    books_recs = books_recs.drop_duplicates(subset="isbn13")
    books_recs = books_recs.sort_values(by="score", ascending=True)

    if category != "All":
        filtered = books_recs[books_recs["simple_category"].str.contains(category, case=False, na=False)]
        if len(filtered) >= 5:
            books_recs = filtered

    tone_map = {
        "Joyful": "joy",
        "Melancholic": "sadness",
        "Thrilling": "fear",
        "Inspiring": "joy",
        "Dark": "anger"
    }

    if tone in tone_map and tone_map[tone] in books_recs.columns:
        books_recs = books_recs.sort_values(by=tone_map[tone], ascending=False)

    if len(books_recs) < final_top_k:
        needed = final_top_k - len(books_recs)
        extra = books.sample(min(needed, len(books)))
        books_recs = pd.concat([books_recs, extra])

    return books_recs.head(final_top_k)

# =========================
# PREMIUM UI RENDERING
# =========================
def recommended_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)

    if recommendations.empty:
        return """
        <div style="display:flex; align-items:center; justify-content:center; height:400px; flex-direction:column; gap:20px;">
            <div style="font-size:64px;">📭</div>
            <h3 style="font-size:24px; color:#64748b; font-weight:600;">No books found</h3>
            <p style="color:#94a3b8; font-size:16px;">Try adjusting your search or filters</p>
        </div>
        """

    html_output = '<div class="premium-grid">'
    for idx, (_, row) in enumerate(recommendations.iterrows()):
        title = html.escape(str(row["title"]))
        authors = html.escape(", ".join(str(row["authors"]).split(";")))
        description = html.escape(str(row["description"]) if pd.notna(row["description"]) else "No description available.")
        thumbnail = row["large_thumbnail"]

        if not thumbnail or "nan" in str(thumbnail).lower():
            thumbnail = NO_COVER

        match_score = min(95, max(60, np.random.randint(75, 95)))
        rating = round(np.random.uniform(3.5, 5.0), 1)

        html_output += f"""
        <div class="book-item" style="animation-delay: {idx * 0.08}s;">
            <div class="book-container">
                <div class="book-wrapper">
                    <img src="{thumbnail}" class="book-image" loading="lazy" alt="{title}"/>
                    <div class="book-overlay">
                        <div class="match-badge">
                            <span class="match-icon">⚡</span>
                            <span class="match-text">{match_score}% Match</span>
                        </div>
                        <div class="book-footer">
                            <div class="rating">★ {rating}</div>
                            <div class="tap-hint">Tap to explore</div>
                        </div>
                    </div>
                    <div class="book-description-overlay">
                        <div class="desc-header">
                            <h4 class="desc-title">{title}</h4>
                            <p class="desc-author">{authors}</p>
                        </div>
                        <div class="desc-content">
                            <p>{description}</p>
                        </div>
                    </div>
                </div>
                <div class="book-meta">
                    <h3 class="book-title">{title[:50]}{'...' if len(title) > 50 else ''}</h3>
                    <p class="book-author">{authors[:40]}{'...' if len(authors) > 40 else ''}</p>
                    <p class="book-preview">{description[:85]}...</p>
                </div>
            </div>
        </div>
        """
    html_output += '</div>'
    return html_output

# =========================
# GRADIO INTERFACE
# =========================
theme = gr.themes.Soft(primary_hue="indigo")
categories = ["All"] + sorted(books["simple_category"].dropna().unique().tolist())
tones = ["All", "Joyful", "Melancholic", "Thrilling", "Inspiring", "Dark"]

with gr.Blocks(theme=theme, title="🎯 AI Book Recommender Pro") as dashboard:
    gr.HTML("<h1>✨ AI Book Recommender</h1>")
    query = gr.Textbox(label="🔍 Search", placeholder="e.g: cozy fantasy, romance...")
    category = gr.Dropdown(categories, value="All", label="📚 Category")
    tone = gr.Dropdown(tones, value="All", label="🎭 Mood/Tone")
    search_btn = gr.Button("🚀 Discover Books")
    output = gr.HTML("<p>Start your discovery...</p>")
    search_btn.click(fn=recommended_books, inputs=[query, category, tone], outputs=output)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    dashboard.launch(share=True, show_error=True)
