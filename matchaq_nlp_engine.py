import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fitz
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

WEIGHT_SEMANTIC = 0.6
WEIGHT_KEYWORD = 0.4

W_TECH = 0.60
W_LANG = 0.25
W_SOFT = 0.15

THRESHOLD_TH = 0.50
THRESHOLD_EN = 0.54
JD_SEM_ACTIVATION = 0.35
PF_SEM_ACTIVATION = 0.40
SOFT_MATCH_SEM_THRESHOLD = 45.0

NON_WORD_PATTERN = re.compile(r"[^\w\s]+", re.UNICODE)
SPACE_PATTERN = re.compile(r"\s+")
LATIN_OR_DIGIT_PATTERN = re.compile(r"[A-Za-z0-9]")
THAI_CHAR_PATTERN = re.compile(r"[ก-๙]")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
HSK_PATTERN = re.compile(r"hsk\s*(?:level\s*)?([1-6])", re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"[\n.!?;]+")

SKILL_ALIASES = {
    "python": ["python", "py", "ไพธอน", "ภาษาไพธอน"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite", "เอสคิวแอล"],
    "machine learning": ["machine learning", "ml", "การเรียนรู้ของเครื่อง", "แมชชีนเลิร์นนิง"],
    "deep learning": ["deep learning", "dl", "neural network", "การเรียนรู้เชิงลึก", "โครงข่ายประสาทเทียม"],
    "data analysis": ["data analysis", "data analytics", "วิเคราะห์ข้อมูล", "การวิเคราะห์ข้อมูล"],
    "nlp": ["nlp", "natural language processing", "ประมวลผลภาษาธรรมชาติ"],
    "tensorflow": ["tensorflow", "tf", "เทนเซอร์โฟลว์"],
    "pytorch": ["pytorch", "torch", "ไพทอร์ช"],
    "scikit-learn": ["scikit-learn", "sklearn", "ไซคิตเลิร์น"],
    "pandas": ["pandas", "แพนดาส"],
    "numpy": ["numpy", "np", "นัมไพ"],
    "excel": ["excel", "microsoft excel", "เอ็กเซล"],
    "power bi": ["power bi", "powerbi", "พาวเวอร์บีไอ"],
    "tableau": ["tableau", "แท็บโล"],
    "git": ["git", "github", "gitlab", "กิต", "กิทฮับ"],
    "docker": ["docker", "container", "ด็อกเกอร์", "คอนเทนเนอร์"],
    "chinese language": ["chinese", "mandarin", "ภาษาจีน", "จีนกลาง", "中文", "汉语"],
    "hsk certification": ["hsk", "hsk level", "ผลสอบ hsk", "คะแนน hsk", "汉语水平考试"],
    "chinese technical reading": [
        "product specification sheets in chinese",
        "understand product specification sheets in chinese",
        "อ่านสเปกสินค้าเป็นภาษาจีน",
        "อ่านเอกสารเทคนิคภาษาจีน",
    ],
    "chinese executive reporting": [
        "report to chinese-speaking executives",
        "present to chinese-speaking executives",
        "รายงานผู้บริหารที่ใช้ภาษาจีน",
        "นำเสนอผู้บริหารภาษาจีน",
    ],
    "communication": [
        "communication",
        "communicate",
        "communication skills",
        "การสื่อสาร",
        "ทักษะการสื่อสาร",
        "สื่อสารได้ดี",
    ],
    "problem solving": [
        "problem solving",
        "problem-solving",
        "solve problems",
        "การแก้ปัญหา",
        "แก้ปัญหา",
        "วิเคราะห์ปัญหา",
    ],
    "teamwork": [
        "teamwork",
        "team work",
        "collaboration",
        "collaborative",
        "ทำงานเป็นทีม",
        "การทำงานร่วมกัน",
        "ร่วมงานกับทีม",
    ],
    "leadership": [
        "leadership",
        "lead",
        "team lead",
        "ภาวะผู้นำ",
        "การเป็นผู้นำ",
        "นำทีม",
    ],
    "time management": [
        "time management",
        "prioritization",
        "deadline management",
        "บริหารเวลา",
        "จัดลำดับความสำคัญ",
        "ตรงต่อเวลา",
    ],
    "proactiveness": [
        "proactive",
        "proactiveness",
        "initiative",
        "self-starter",
        "กระตือรือร้น",
        "มีความกระตือรือร้น",
        "มีความริเริ่ม",
        "เชิงรุก",
    ],
    "creativity": [
        "creativity",
        "creative thinking",
        "innovative",
        "ความคิดสร้างสรรค์",
        "คิดสร้างสรรค์",
        "นวัตกรรม",
    ],
    "adaptability": [
        "adaptability",
        "adaptable",
        "flexible",
        "learning agility",
        "ปรับตัวได้ดี",
        "ยืดหยุ่น",
        "เรียนรู้เร็ว",
    ],
    "critical thinking": [
        "critical thinking",
        "analytical thinking",
        "logical thinking",
        "คิดเชิงวิเคราะห์",
        "คิดอย่างมีเหตุผล",
        "วิจารณญาณ",
    ],
    "responsibility": [
        "responsibility",
        "accountability",
        "ownership",
        "reliable",
        "ความรับผิดชอบ",
        "รับผิดชอบ",
        "มีวินัย",
    ],
    "attention to detail": [
        "attention to detail",
        "detail-oriented",
        "accuracy",
        "ใส่ใจรายละเอียด",
        "ละเอียดรอบคอบ",
        "ความถูกต้อง",
    ],
    "presentation": [
        "presentation",
        "presenting",
        "public speaking",
        "การนำเสนอ",
        "ทักษะการนำเสนอ",
        "พูดในที่สาธารณะ",
    ],
    "negotiation": [
        "negotiation",
        "negotiating",
        "influencing",
        "การเจรจาต่อรอง",
        "ทักษะการเจรจา",
        "โน้มน้าว",
    ],
}

SKILL_INTENTS = {
    "python": [
        "can write python scripts for automation and data processing",
        "developed projects using python programming",
        "สามารถเขียนโปรแกรมภาษาไพธอนเพื่อทำงานอัตโนมัติและวิเคราะห์ข้อมูล",
    ],
    "sql": [
        "can query and manage relational databases with sql",
        "writes joins and aggregation queries",
        "สามารถเขียนคำสั่ง sql เพื่อดึงและจัดการข้อมูลจากฐานข้อมูล",
    ],
    "machine learning": [
        "builds and evaluates predictive machine learning models",
        "applies supervised and unsupervised learning methods",
        "สามารถสร้างและประเมินโมเดลการเรียนรู้ของเครื่อง",
    ],
    "deep learning": [
        "uses neural networks for model development",
        "builds deep learning solutions with modern frameworks",
        "สามารถพัฒนาโมเดลโครงข่ายประสาทเทียม",
    ],
    "data analysis": [
        "analyzes datasets to extract insights for decisions",
        "creates analysis reports from data",
        "สามารถวิเคราะห์ข้อมูลและสรุปข้อมูลเชิงลึก",
    ],
    "nlp": [
        "works on natural language processing tasks",
        "builds text classification and language understanding models",
        "สามารถพัฒนางานประมวลผลภาษาธรรมชาติ",
    ],
    "tensorflow": [
        "uses tensorflow for model training and deployment",
        "พัฒนาโมเดลด้วย tensorflow",
    ],
    "pytorch": [
        "uses pytorch for deep learning experiments",
        "พัฒนาโมเดลด้วย pytorch",
    ],
    "scikit-learn": [
        "uses scikit learn for machine learning pipelines",
        "ใช้งาน scikit learn สำหรับงาน machine learning",
    ],
    "pandas": [
        "uses pandas for data cleaning and transformation",
        "ใช้ pandas จัดการและเตรียมข้อมูล",
    ],
    "numpy": [
        "uses numpy for numerical computing",
        "ใช้ numpy สำหรับคำนวณเชิงตัวเลข",
    ],
    "excel": [
        "uses excel for data reporting and analysis",
        "ใช้ excel สรุปรายงานและวิเคราะห์ข้อมูล",
    ],
    "power bi": [
        "builds dashboard in power bi",
        "สร้าง dashboard ด้วย power bi",
    ],
    "tableau": [
        "creates interactive dashboard in tableau",
        "สร้างแดชบอร์ดด้วย tableau",
    ],
    "git": [
        "uses git for version control and collaboration",
        "ใช้ git จัดการเวอร์ชันและทำงานร่วมกับทีม",
    ],
    "docker": [
        "uses docker to containerize applications",
        "ใช้ docker ในการทำ container แอปพลิเคชัน",
    ],
    "chinese language": [
        "can communicate professionally in chinese",
        "can work with chinese speaking stakeholders",
        "สามารถสื่อสารภาษาจีนในการทำงานได้",
    ],
    "hsk certification": [
        "has hsk certification proving chinese proficiency",
        "passed chinese proficiency test hsk",
        "มีผลสอบ hsk เพื่อยืนยันความสามารถภาษาจีน",
    ],
    "chinese technical reading": [
        "can deeply understand product specification sheets in chinese",
        "can interpret chinese technical documents accurately",
        "สามารถอ่านและเข้าใจเอกสารเทคนิคภาษาจีนได้อย่างลึกซึ้ง",
    ],
    "chinese executive reporting": [
        "can report progress to chinese speaking executives",
        "can present updates in chinese to management",
        "สามารถรายงานงานให้ผู้บริหารที่ใช้ภาษาจีนได้",
    ],
    "communication": [
        "can communicate technical findings clearly to stakeholders",
        "presents results in simple business language",
        "สามารถอธิบายประเด็นทางเทคนิคให้ผู้เกี่ยวข้องเข้าใจง่าย",
    ],
    "problem solving": [
        "solves complex issues with structured reasoning",
        "identifies root causes and proposes practical solutions",
        "สามารถวิเคราะห์สาเหตุและแก้ปัญหาอย่างเป็นระบบ",
    ],
    "teamwork": [
        "collaborates effectively in cross functional teams",
        "works well with others to deliver shared goals",
        "สามารถทำงานร่วมกับทีมข้ามสายงานได้ดี",
    ],
    "leadership": [
        "can lead projects and coordinate team execution",
        "takes ownership and drives outcomes",
        "สามารถนำทีมและขับเคลื่อนงานให้สำเร็จ",
    ],
    "time management": [
        "can prioritize tasks and manage deadlines effectively",
        "delivers work on time under constraints",
        "สามารถบริหารเวลาและลำดับความสำคัญงานได้ดี",
    ],
    "proactiveness": [
        "takes initiative without being asked",
        "actively proposes improvements and actions",
        "มีความกระตือรือร้นและเริ่มลงมือทำเอง",
    ],
    "creativity": [
        "generates creative ideas and innovative solutions",
        "thinks outside the box for better outcomes",
        "มีความคิดสร้างสรรค์และเสนอแนวทางใหม่",
    ],
    "adaptability": [
        "adapts quickly to changes and new environments",
        "learns new tools and methods rapidly",
        "ปรับตัวได้ดีเมื่อมีการเปลี่ยนแปลง",
    ],
    "critical thinking": [
        "evaluates information logically before making decisions",
        "uses evidence based reasoning to assess options",
        "คิดเชิงวิเคราะห์และตัดสินใจอย่างมีเหตุผล",
    ],
    "responsibility": [
        "takes ownership and accountability for tasks",
        "delivers reliable and dependable work",
        "มีความรับผิดชอบและเชื่อถือได้",
    ],
    "attention to detail": [
        "works carefully with high attention to detail",
        "maintains accuracy and quality in deliverables",
        "ใส่ใจรายละเอียดและความถูกต้องของงาน",
    ],
    "presentation": [
        "can present ideas clearly to audience and management",
        "delivers structured presentations effectively",
        "สามารถนำเสนอข้อมูลอย่างชัดเจน",
    ],
    "negotiation": [
        "can negotiate and align stakeholders effectively",
        "handles conflicts and reaches win win agreements",
        "สามารถเจรจาต่อรองและประสานผลประโยชน์ได้ดี",
    ],
}

SOFT_INTENTS = {skill: SKILL_INTENTS[skill] for skill in [
    "communication",
    "problem solving",
    "teamwork",
    "leadership",
    "time management",
    "proactiveness",
    "creativity",
    "adaptability",
    "critical thinking",
    "responsibility",
    "attention to detail",
    "presentation",
    "negotiation",
]}

SOFT_SKILLS = {
    "communication",
    "problem solving",
    "teamwork",
    "leadership",
    "time management",
    "proactiveness",
    "creativity",
    "adaptability",
    "critical thinking",
    "responsibility",
    "attention to detail",
    "presentation",
    "negotiation",
}

TECH_SKILLS = {
    "python",
    "sql",
    "machine learning",
    "deep learning",
    "data analysis",
    "nlp",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "excel",
    "power bi",
    "tableau",
    "git",
    "docker",
}

LANG_SKILLS = {
    "chinese language",
    "hsk certification",
    "chinese technical reading",
    "chinese executive reporting",
}

SKILL_WEIGHTS = {
    "python": 1.2,
    "sql": 1.1,
    "machine learning": 1.2,
    "deep learning": 1.2,
    "tensorflow": 1.1,
    "pytorch": 1.1,
    "scikit-learn": 1.1,
    "pandas": 1.1,
    "numpy": 1.1,
    "chinese language": 1.2,
    "hsk certification": 1.6,
    "communication": 1.1,
    "problem solving": 1.1,
    "teamwork": 1.05,
    "leadership": 1.05,
    "time management": 1.05,
    "proactiveness": 1.05,
    "creativity": 1.05,
    "adaptability": 1.05,
    "critical thinking": 1.1,
    "responsibility": 1.05,
    "attention to detail": 1.05,
    "presentation": 1.05,
    "negotiation": 1.05,
}
DEFAULT_SKILL_WEIGHT = 1.0


@dataclass
class SkillMatch:
    alias_hits: List[str]
    intent_hits: List[str]
    score: float


@dataclass
class MatchResult:
    portfolio: str
    semantic_score: float
    keyword_score: float
    final_score: float
    coverage_score: float
    hsk_level: int
    matched_tech_skills: List[str]
    matched_language_skills: List[str]
    matched_soft_skills: List[str]
    soft_skill_evidence_sentences: List[str]
    recommendation: str


def normalize_text(text: str) -> str:
    text = NON_WORD_PATTERN.sub(" ", text.lower())
    return SPACE_PATTERN.sub(" ", text).strip()


def compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9ก-๙]+", "", text.lower())


def split_sentences(text: str) -> List[str]:
    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if len(part.strip().split()) >= 4]
    return sentences if sentences else [text.strip()]


def _needs_substring_match(term: str) -> bool:
    return bool(THAI_CHAR_PATTERN.search(term) or CJK_PATTERN.search(term))


def thai_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(THAI_CHAR_PATTERN.findall(text)) / max(1, len(text))


def adaptive_threshold(text: str, th_th: float = THRESHOLD_TH, th_en: float = THRESHOLD_EN) -> float:
    ratio = thai_ratio(text)
    return (ratio * th_th) + ((1 - ratio) * th_en)


def _contains_term(normalized_text: str, token_set: set[str], term: str) -> bool:
    normalized_term = normalize_text(term)
    if not normalized_term:
        return False
    if _needs_substring_match(normalized_term):
        return normalized_term in normalized_text
    if " " in normalized_term:
        return normalized_term in normalized_text
    if len(normalized_term) <= 2:
        return normalized_term in token_set
    if LATIN_OR_DIGIT_PATTERN.search(normalized_term):
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])"
        return re.search(pattern, normalized_text) is not None
    return normalized_term in normalized_text


def _match_skill(normalized_text: str, token_set: set[str], skill: str) -> SkillMatch:
    alias_hits = []
    intent_hits = []

    for alias in SKILL_ALIASES.get(skill, []):
        if _contains_term(normalized_text, token_set, alias):
            alias_hits.append(alias)

    for intent in SKILL_INTENTS.get(skill, []):
        if _contains_term(normalized_text, token_set, intent):
            intent_hits.append(intent)

    score = 0.0
    if alias_hits:
        score += 1.0
    if intent_hits:
        score += 0.7
    return SkillMatch(alias_hits=alias_hits, intent_hits=intent_hits, score=min(score, 1.0))


def extract_skill_map(text: str) -> Dict[str, SkillMatch]:
    normalized_text = normalize_text(text)
    token_set = set(normalized_text.split())
    skill_map: Dict[str, SkillMatch] = {}

    for skill in SKILL_ALIASES:
        match = _match_skill(normalized_text, token_set, skill)
        if match.score > 0:
            skill_map[skill] = match

    return skill_map


def extract_keywords_from_text(text: str) -> List[str]:
    norm = normalize_text(text)
    comp = compact_text(text)
    no_space = norm.replace(" ", "")
    found = []

    for skill, aliases in SKILL_ALIASES.items():
        hit = False
        for alias in aliases:
            a_norm = normalize_text(alias)
            a_comp = compact_text(alias)

            if a_norm and re.search(rf"(?<!\w){re.escape(a_norm)}(?!\w)", norm):
                hit = True
                break
            if len(a_comp) >= 4 and a_comp in comp:
                hit = True
                break
            if len(a_comp) >= 4 and a_comp in no_space:
                hit = True
                break
        if hit:
            found.append(skill)

    return sorted(set(found))


def categorize_skills(skills: Iterable[str]) -> tuple[List[str], List[str], List[str]]:
    ordered = sorted(set(skills))
    tech = [skill for skill in ordered if skill in TECH_SKILLS]
    lang = [skill for skill in ordered if skill in LANG_SKILLS]
    soft = [skill for skill in ordered if skill in SOFT_SKILLS]
    return tech, lang, soft


def extract_match_evidence(text: str, matched_skills: Sequence[str]) -> List[str]:
    if not matched_skills:
        return []

    sentences = split_sentences(text)
    evidence: List[str] = []
    for sentence in sentences:
        normalized_sentence = normalize_text(sentence)
        token_set = set(normalized_sentence.split())
        for skill in matched_skills:
            if any(_contains_term(normalized_sentence, token_set, alias) for alias in SKILL_ALIASES.get(skill, [])):
                evidence.append(sentence.strip())
                break
            if any(_contains_term(normalized_sentence, token_set, intent) for intent in SKILL_INTENTS.get(skill, [])):
                evidence.append(sentence.strip())
                break

    unique_evidence: List[str] = []
    seen = set()
    for sentence in evidence:
        if sentence and sentence not in seen:
            unique_evidence.append(sentence)
            seen.add(sentence)
    return unique_evidence[:3]


def extract_text_from_pdf(path: str) -> str:
    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def detect_hsk(text: str) -> int:
    levels = [int(level) for level in HSK_PATTERN.findall(text)]
    return max(levels) if levels else 0


@lru_cache(maxsize=1)
def get_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            local_files_only=True,
        )
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_intent_embeddings() -> Dict[str, np.ndarray]:
    model = get_model()
    if model is None:
        return {}
    return {skill: np.array(model.encode(intents, normalize_embeddings=True)) for skill, intents in SKILL_INTENTS.items()}


@lru_cache(maxsize=1)
def get_soft_intent_embeddings() -> Dict[str, np.ndarray]:
    model = get_model()
    if model is None:
        return {}
    return {skill: np.array(model.encode(intents, normalize_embeddings=True)) for skill, intents in SOFT_INTENTS.items()}


def semantic_batch(jd_text: str, portfolio_texts: Sequence[str]) -> List[float]:
    model = get_model()
    if model is not None:
        embeddings = model.encode([jd_text] + list(portfolio_texts), normalize_embeddings=True)
        similarities = cosine_similarity([embeddings[0]], embeddings[1:])[0]
        return [round(float(score * 100), 2) for score in similarities]

    docs = [jd_text] + list(portfolio_texts)
    matrix = TfidfVectorizer().fit_transform(docs)
    similarities = cosine_similarity(matrix[0:1], matrix[1:])[0]
    return [round(float(score * 100), 2) for score in similarities]


def semantic_skill_scores(text: str, th_th: float = THRESHOLD_TH, th_en: float = THRESHOLD_EN) -> Dict[str, float]:
    sentences = split_sentences(text)
    if not sentences:
        return {}

    model = get_model()
    intent_embeddings = get_intent_embeddings()
    if model is None or not intent_embeddings:
        exact_skills = set(extract_keywords_from_text(text))
        return {skill: 1.0 for skill in exact_skills}

    sentence_vectors = np.array(model.encode(sentences, normalize_embeddings=True))
    threshold = adaptive_threshold(text, th_th, th_en)

    scores = {}
    for skill, proto_vectors in intent_embeddings.items():
        similarity = float(cosine_similarity(sentence_vectors, proto_vectors).max())
        scores[skill] = 0.0 if similarity < threshold else (similarity - threshold) / (1 - threshold)
    return scores


def score_soft_skills_with_evidence(text: str, top_k: int = 2) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    sentences = split_sentences(text)
    if not sentences:
        return {}, {}

    model = get_model()
    soft_embeddings = get_soft_intent_embeddings()
    if model is None or not soft_embeddings:
        fallback_skills = [skill for skill in extract_keywords_from_text(text) if skill in SOFT_SKILLS]
        evidence = extract_match_evidence(text, fallback_skills)
        return {skill: 100.0 for skill in fallback_skills}, {skill: evidence for skill in fallback_skills}

    sentence_vectors = np.array(model.encode(sentences, normalize_embeddings=True))
    threshold = adaptive_threshold(text, th_th=THRESHOLD_TH, th_en=THRESHOLD_EN)

    soft_scores: Dict[str, float] = {}
    soft_evidence: Dict[str, List[str]] = {}

    for skill, proto_vectors in soft_embeddings.items():
        similarity_matrix = cosine_similarity(sentence_vectors, proto_vectors)
        best_per_sentence = similarity_matrix.max(axis=1)
        best = float(best_per_sentence.max())

        if best < threshold:
            soft_scores[skill] = 0.0
            soft_evidence[skill] = []
            continue

        soft_scores[skill] = round(min(100.0, ((best - threshold) / (1 - threshold)) * 100), 2)
        indices = np.argsort(best_per_sentence)[::-1][:top_k]
        soft_evidence[skill] = [sentences[i] for i in indices if (best_per_sentence[i] * 100) >= SOFT_MATCH_SEM_THRESHOLD]

    return soft_scores, soft_evidence


def weighted_blended_group_score(
    jd_group: set[str],
    pf_exact_set: set[str],
    pf_sem_scores: Dict[str, float],
) -> Optional[float]:
    if not jd_group:
        return None

    numerator = 0.0
    denominator = 0.0
    for skill in jd_group:
        weight = SKILL_WEIGHTS.get(skill, DEFAULT_SKILL_WEIGHT)
        denominator += weight
        hit = max(1.0 if skill in pf_exact_set else 0.0, pf_sem_scores.get(skill, 0.0))
        numerator += weight * hit

    return (numerator / denominator) * 100 if denominator > 0 else 0.0


def keyword_match_score_3groups(
    jd_keywords: Sequence[str],
    pf_keywords: Sequence[str],
    pf_text: str,
    jd_text: str,
) -> Tuple[float, int, List[str], Dict[str, List[str]]]:
    jd_exact = set(jd_keywords)
    pf_exact = set(pf_keywords)

    jd_sem = semantic_skill_scores(jd_text, th_th=0.49, th_en=0.53)
    pf_sem = semantic_skill_scores(pf_text, th_th=PF_SEM_ACTIVATION, th_en=THRESHOLD_EN)

    jd_all = jd_exact | {skill for skill, score in jd_sem.items() if score >= JD_SEM_ACTIVATION}
    jd_tech = jd_all & TECH_SKILLS
    jd_lang = jd_all & LANG_SKILLS
    jd_soft = jd_all & SOFT_SKILLS

    soft_scores, soft_evidence = score_soft_skills_with_evidence(pf_text)
    for skill, score in soft_scores.items():
        pf_sem[skill] = max(pf_sem.get(skill, 0.0), score / 100.0)

    tech_score = weighted_blended_group_score(jd_tech, pf_exact, pf_sem)
    lang_score = weighted_blended_group_score(jd_lang, pf_exact, pf_sem)
    soft_score = weighted_blended_group_score(jd_soft, pf_exact, pf_sem)

    hsk_level = detect_hsk(pf_text)
    if lang_score is not None:
        if hsk_level >= 5:
            lang_score = min(100.0, lang_score + 8.0)
        elif hsk_level == 4:
            lang_score = min(100.0, lang_score + 6.0)
        elif hsk_level >= 1:
            lang_score = min(100.0, lang_score + 2.0)

    scores = []
    weights = []
    if tech_score is not None:
        scores.append(tech_score)
        weights.append(W_TECH)
    if lang_score is not None:
        scores.append(lang_score)
        weights.append(W_LANG)
    if soft_score is not None:
        scores.append(soft_score)
        weights.append(W_SOFT)

    if not scores:
        return 0.0, hsk_level, [], soft_evidence

    total_weight = sum(weights)
    keyword_score = sum(score * (weight / total_weight) for score, weight in zip(scores, weights))

    matched_skills = []
    for skill in jd_all:
        if skill in pf_exact or pf_sem.get(skill, 0.0) >= PF_SEM_ACTIVATION:
            matched_skills.append(skill)

    return round(keyword_score, 2), hsk_level, sorted(set(matched_skills)), soft_evidence


def coverage_score_from_matches(jd_keywords: Sequence[str], matched_skills: Sequence[str], jd_text: str) -> float:
    jd_exact = set(jd_keywords)
    jd_sem = semantic_skill_scores(jd_text, th_th=0.49, th_en=0.53)
    jd_all = jd_exact | {skill for skill, score in jd_sem.items() if score >= JD_SEM_ACTIVATION}
    if not jd_all:
        return 0.0
    return round((len(set(matched_skills) & jd_all) / len(jd_all)) * 100, 2)


def recommendation_label(score: float) -> str:
    if score >= 55:
        return "น่าสัมภาษณ์"
    return "ยังไม่ตรงตามเงื่อนไข"


def evaluate_portfolios(
    jd_path: str,
    portfolio_paths: Sequence[str],
    portfolio_names: Optional[Sequence[str]] = None,
) -> List[MatchResult]:
    jd_text = extract_text_from_pdf(jd_path)
    pf_texts = [extract_text_from_pdf(path) for path in portfolio_paths]
    semantic_scores = semantic_batch(jd_text, pf_texts)
    jd_keywords = extract_keywords_from_text(jd_text)

    names = list(portfolio_names) if portfolio_names is not None else [Path(path).name for path in portfolio_paths]
    results: List[MatchResult] = []

    for path, display_name, pf_text, semantic_score in zip(portfolio_paths, names, pf_texts, semantic_scores):
        pf_keywords = extract_keywords_from_text(pf_text)
        keyword_score, hsk_level, matched_skills, soft_evidence_map = keyword_match_score_3groups(
            jd_keywords,
            pf_keywords,
            pf_text,
            jd_text,
        )
        coverage_score = coverage_score_from_matches(jd_keywords, matched_skills, jd_text)
        final_score = round(
            (semantic_score * WEIGHT_SEMANTIC) + (keyword_score * WEIGHT_KEYWORD),
            2,
        )

        matched_tech, matched_lang, matched_soft = categorize_skills(matched_skills)
        evidence_skills = matched_soft or matched_skills
        soft_evidence = []
        for skill in matched_soft:
            soft_evidence.extend(soft_evidence_map.get(skill, []))
        if not soft_evidence:
            soft_evidence = extract_match_evidence(pf_text, evidence_skills)
        else:
            unique_evidence = []
            seen = set()
            for sentence in soft_evidence:
                if sentence not in seen:
                    unique_evidence.append(sentence)
                    seen.add(sentence)
            soft_evidence = unique_evidence[:3]

        results.append(
            MatchResult(
                portfolio=display_name or Path(path).name,
                semantic_score=round(semantic_score, 2),
                keyword_score=round(keyword_score, 2),
                final_score=final_score,
                coverage_score=coverage_score,
                hsk_level=hsk_level,
                matched_tech_skills=matched_tech,
                matched_language_skills=matched_lang,
                matched_soft_skills=matched_soft,
                soft_skill_evidence_sentences=soft_evidence,
                recommendation=recommendation_label(final_score),
            )
        )

    return sorted(results, key=lambda item: item.final_score, reverse=True)


def results_to_df(results: Sequence[MatchResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "Portfolio": result.portfolio,
                "Semantic Similarity (%)": result.semantic_score,
                "Keyword Match (%)": result.keyword_score,
                "Overall Score (%)": result.final_score,
                "Skill Coverage (%)": result.coverage_score,
                "HSK Level": result.hsk_level if result.hsk_level > 0 else "-",
                "Recommendation": result.recommendation,
                "Matched JD Technical Skills": ", ".join(result.matched_tech_skills) if result.matched_tech_skills else "-",
                "Matched JD Language Skills": ", ".join(result.matched_language_skills) if result.matched_language_skills else "-",
                "Matched JD Soft Skills": ", ".join(result.matched_soft_skills) if result.matched_soft_skills else "-",
                "Soft Skill Evidence Sentences": " | ".join(result.soft_skill_evidence_sentences)
                if result.soft_skill_evidence_sentences
                else "-",
            }
        )
    return pd.DataFrame(rows)
