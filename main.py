from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

import shutil
import os
import uuid

from nlp_engine import evaluate_portfolios, results_to_df

app = FastAPI()

# ✅ static (สำคัญ)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_file(upload_file: UploadFile):
    file_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, file_id + "_" + upload_file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return path


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/analyze")
def analyze_redirect():
    return RedirectResponse(url="/", status_code=303)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, jd: UploadFile = File(...), portfolios: list[UploadFile] = File(...)):
    
    jd_path = save_file(jd)
    portfolio_paths = [save_file(p) for p in portfolios]
    portfolio_names = [p.filename for p in portfolios]

    # ✅ NLP
    results = evaluate_portfolios(jd_path, portfolio_paths, portfolio_names=portfolio_names)
    df = results_to_df(results)

    # 🔥 แก้ data mapping ให้ HTML ใช้ได้
    data = []

    for row in df.fillna("-").to_dict(orient="records"):
        data.append({
            "Portfolio": row.get("Portfolio", "-"),
            "overall": row.get("Overall Score (%)", "-"),
            "semantic": row.get("Semantic Similarity (%)", "-"),
            "keyword": row.get("Keyword Match (%)", "-"),
            "coverage": row.get("Skill Coverage (%)", "-"),
            "hsk": row.get("HSK Level", "-"),
            "recommendation": row.get("Recommendation", "-"),

            # ✅ skill
            "tech": row.get("Matched JD Technical Skills", "").split(", ") if row.get("Matched JD Technical Skills") not in ["-", None] else [],
            "language": row.get("Matched JD Language Skills", "").split(", ") if row.get("Matched JD Language Skills") not in ["-", None] else [],
            "soft": row.get("Matched JD Soft Skills", "").split(", ") if row.get("Matched JD Soft Skills") not in ["-", None] else [],
            "evidence": row.get("Soft Skill Evidence Sentences", "-"),
        })

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "results": data
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
