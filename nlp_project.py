import fitz  # pymupdf
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# โหลด model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


# =========================
# 1) Extract text
# =========================
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.lower()


# =========================
# 2) Keyword simple matching
# =========================
def keyword_score(jd_text, pf_text):
    jd_words = set(re.findall(r'\w+', jd_text))
    pf_words = set(re.findall(r'\w+', pf_text))

    if len(jd_words) == 0:
        return 0

    match = jd_words.intersection(pf_words)
    score = (len(match) / len(jd_words)) * 100
    return round(score, 2)


# =========================
# 3) Evaluate portfolios
# =========================
def evaluate_portfolios(jd_path, portfolio_paths):
    jd_text = extract_text_from_pdf(jd_path)

    texts = [jd_text]
    pf_texts = []

    for p in portfolio_paths:
        text = extract_text_from_pdf(p)
        pf_texts.append(text)
        texts.append(text)

    embeddings = model.encode(texts)

    jd_emb = embeddings[0]

    results = []

    for i, path in enumerate(portfolio_paths):
        pf_emb = embeddings[i + 1]

        # semantic score
        semantic = cosine_similarity([jd_emb], [pf_emb])[0][0] * 100

        # keyword score
        keyword = keyword_score(jd_text, pf_texts[i])

        # final score (combine)
        final = (semantic * 0.6) + (keyword * 0.4)

        results.append({
            "portfolio": path.split("/")[-1],
            "semantic_score": round(semantic, 2),
            "keyword_score": round(keyword, 2),
            "final_score": round(final, 2),
            "hsk_level": "-"
        })

    # sort ranking
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    return results


# =========================
# 4) Convert to DataFrame
# =========================
def results_to_df(results, jd_path, portfolio_paths):
    rows = []

    for r in results:
        score = r["final_score"]

        # recommendation logic
        if score >= 65:
            rec = "เหมาะสมมาก"
        elif score >= 50:
            rec = "น่าสัมภาษณ์"
        elif score >= 40:
            rec = "พิจารณาเพิ่มเติม"
        else:
            rec = "ยังไม่ตรงตำแหน่ง"

        rows.append({
            "Portfolio": r["portfolio"],
            "Semantic Similarity (%)": r["semantic_score"],
            "Keyword Match (%)": r["keyword_score"],
            "Overall Score (%)": r["final_score"],
            "Skill Coverage (%)": r["keyword_score"],  # ใช้แทนก่อน
            "HSK Level": r["hsk_level"],
            "Recommendation": rec
        })

    df = pd.DataFrame(rows).sort_values("Overall Score (%)", ascending=False).reset_index(drop=True)

    return df, None, None, None, None