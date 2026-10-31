"""
AI Resume & Skill Development Assistant (generic version)

Works for any role or industry. Pipeline:
  1. Extract requirements from the job description (JD) with plain-text rules.
  2. Pull the key terms out of each requirement (no hardcoded skill list).
  3. Match those terms against the resume with light stemming + a few synonyms.
  4. Suggest practice tasks and free learning links per gap.
  5. Optionally, a local LLM (Ollama) writes a four-week plan.

Run:  streamlit run app_genai_resume_assistant_generic.py
"""
import datetime
import hashlib
import re
from urllib.parse import quote_plus

MODEL_NAME = "llama3.2:3b"

CLEAR = "Clearly Evidenced"
PARTIAL = "Partially Evidenced"
NONE = "Not Evidenced"
MANUAL = "Needs Manual Check"
STATUSES = [CLEAR, PARTIAL, MANUAL, NONE]

# (best-sentence score for CLEAR, overall coverage for CLEAR, coverage for PARTIAL)
STRICTNESS = {
    "Lenient": (0.50, 0.75, 0.25),
    "Balanced": (0.60, 0.85, 0.34),
    "Strict": (0.75, 0.95, 0.50),
}

# ----------------------------------------------------------------------------
# Text utilities
# ----------------------------------------------------------------------------
STOPWORDS = set("""
a an the and or of to in on for with from by at as is are be been being this that these those it its into
your you our we their they them will would can could should must may might shall do does did have has had
not no nor than then so such via per etc e.g i.e using use used including include includes within across
among over under about above below between through while where when which who whom whose what how all any
each both either neither other another more most some many much few several only own same too very just
also well able ability abilities strong good great excellent solid proven demonstrated demonstrable
knowledge understanding familiarity familiar proficiency proficient skill skills experience experienced
required requirement requirements preferred preferably desirable desired plus minimum least relevant
responsible responsibility responsibilities candidate candidates role position job work working works
worked ensure ensuring support supporting help helping year years month months new based related
strongly highly high sound clear detailed closely obtain particularly various key field
discipline professional part effective others comprehensive extensive thorough
""".split())

# Words that mean the same thing for matching purposes. Keep this list short and safe.
GROUPS = {
    "grp_analysis": ["analyze", "analyse", "analysis", "analyst", "analytics", "analytical", "analyzing"],
    "grp_manage": ["manage", "manager", "management", "managing", "supervise", "supervisor", "supervision", "oversee"],
    "grp_lead": ["lead", "leader", "leadership", "leading", "led"],
    "grp_communication": ["communicate", "communication", "communications", "communicator", "present", "presented", "presenting", "presentation", "presentations", "briefed", "briefing"],
    "grp_collab": ["collaborate", "collaboration", "collaborative", "teamwork", "cooperate", "cooperation"],
    "grp_report": ["report", "reporting", "dashboard", "dashboards"],
    "grp_customer": ["customer", "customers", "client", "clients"],
    "grp_problem": ["problem", "troubleshoot", "troubleshooting", "debug", "debugging"],
    "grp_develop": ["develop", "developer", "development", "developing"],
    "grp_javascript": ["javascript", "js"],
    "grp_kubernetes": ["kubernetes", "k8s"],
    "grp_postgres": ["postgres", "postgresql"],
    "grp_marketing": ["marketing", "marketer"],
    "edu_bachelor": ["bachelor", "bachelors", "bs", "bsc", "bba", "btech", "beng", "undergraduate", "degree",
                      "b.com", "bcom", "b.sc", "b.tech", "b.s", "b.a", "b.e", "b.b.a"],
    "edu_master": ["master", "masters", "msc", "mba", "postgraduate", "m.sc", "mcom", "mphil", "m.s", "m.a", "m.b.a"],
    "edu_phd": ["phd", "doctorate", "doctoral"],
}

BRITISH = [
    ("organis", "organiz"), ("prioritis", "prioritiz"), ("summaris", "summariz"), ("optimis", "optimiz"),
    ("standardis", "standardiz"), ("utilis", "utiliz"), ("recognis", "recogniz"), ("specialis", "specializ"),
    ("minimis", "minimiz"), ("maximis", "maximiz"), ("visualis", "visualiz"), ("customis", "customiz"),
    ("finalis", "finaliz"), ("categoris", "categoriz"), ("behaviour", "behavior"), ("programme", "program"),
    ("centre", "center"), ("licence", "license"), ("catalogue", "catalog"),
]

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[+#]+|\.[A-Za-z0-9]+)*")


def tokenize(text):
    return TOKEN_RE.findall(text or "")


def _raw_stem(word):
    """Very light stemmer: plurals, -ing, -ed, -ly, trailing e, doubled consonants."""
    w = word.lower()
    for brit, amer in BRITISH:
        if w.startswith(brit):
            w = amer + w[len(brit):]
            break
    if len(w) > 4:
        if w.endswith("ies"):
            w = w[:-3] + "y"
        elif w.endswith(("sses", "ches", "shes", "xes", "zes")):
            w = w[:-2]
        elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        for suffix in ("ing", "edly", "ed", "ly"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 4:
                w = w[: -len(suffix)]
                if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":
                    w = w[:-1]
                break
        if w.endswith("e") and len(w) > 4:
            w = w[:-1]
    return w


CANON = {}
for _canon, _words in GROUPS.items():
    for _w in _words:
        CANON[_w] = _canon
        CANON[_raw_stem(_w)] = _canon


def stem(word):
    w = word.lower()
    return CANON.get(w) or CANON.get(_raw_stem(w)) or _raw_stem(w)


# Generic lead-in verbs in job ads. They count for less and never form "phrases".
ACTION_VERBS = ["conduct", "prepare", "develop", "assess", "liaise", "obtain", "maintain", "monitor", "review",
                "provide", "deliver", "coordinate", "perform", "execute", "undertake", "contribute", "assist",
                "produce", "create", "build", "draft", "implement"]
ACTION_STEMS = {_raw_stem(w) for w in ACTION_VERBS}
# 'macroeconomic' should also count as 'economic'
COMPOUND_PREFIXES = ("macro", "micro", "multi", "inter", "intra", "cross", "sub", "non", "self", "semi",
                     "anti", "geo", "bio", "cyber")


def normalise_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def stem_set(text):
    """All meaningful stems in a piece of text, with education levels implied upward."""
    stems = set()
    for tok in tokenize(text):
        low = tok.lower()
        if low in STOPWORDS:
            continue
        stems.add(stem(low))
        for prefix in COMPOUND_PREFIXES:
            if low.startswith(prefix) and len(low) - len(prefix) >= 5:
                stems.add(stem(low[len(prefix):]))
                break
    if stems & {"excel", "powerpoint"}:  # using Excel or PowerPoint implies Microsoft Office
        stems.update({"office", "ms", "microsoft"})
    if "sql" in stems:  # SQL work implies relational databases
        stems.update({"relational", stem("database")})
    if "edu_phd" in stems:
        stems.update({"edu_master", "edu_bachelor"})
    if "edu_master" in stems:
        stems.add("edu_bachelor")
    return stems


# ----------------------------------------------------------------------------
# Reading files
# ----------------------------------------------------------------------------
def extract_resume_text(uploaded_file):
    """Extract text from PDF, DOCX (including tables), and TXT resumes."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if filename.endswith(".docx"):
        from docx import Document
        document = Document(uploaded_file)
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(part for part in parts if part.strip())
    if filename.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8-sig", errors="replace")
    raise ValueError("Unsupported resume format. Please upload PDF, DOCX, or TXT.")


# ----------------------------------------------------------------------------
# Job description -> requirements
# ----------------------------------------------------------------------------
BULLET_RE = re.compile(r"^\s*(?:[-*•▪◦●■‣–·➢➤✓✔→\uf0b7\uf0a7]|o(?=\s+[A-Z])|\d{1,2}[.)])\s+(.*\S)\s*$")
META_RE = re.compile(
    r"^(job title|title|position|location|salary|department|employment type|job type|company|reports to|"
    r"deadline|closing date|posted|apply by|grade|contract|shift|vacancies)\s*[:\-–]", re.I)
KEEP_HEADS = (
    "requirement", "qualification", "skill", "responsib", "dut", "experience", "competenc", "must have",
    "nice to have", "preferred", "what you", "you will", "you'll", "you have", "you bring", "looking for",
    "education", "essential", "desirable", "key task", "your role", "about you",
)
DROP_HEADS = (
    "about us", "about the company", "who we are", "benefit", "we offer", "what we offer", "perks",
    "compensation", "why join", "why work", "equal opportunity", "how to apply", "application", "culture",
    "our values", "salary", "location",
)
DROP_STARTS = ("what we offer", "we offer", "why join", "why work", "about us", "who we are", "how to apply",
               "equal opportunity", "perks", "benefits")
CUE_WORDS = (
    "experience", "proficien", "knowledge", "ability", "skilled", "familiar", "degree", "years",
    "responsible", "manage", "must", "required", "strong", "understanding", "certif",
)


SMALL_WORDS = {"and", "or", "of", "the", "in", "for", "to", "a", "an", "on", "at", "with", "&"}


def _title_like(text):
    return all(w[:1].isupper() or w.lower() in SMALL_WORDS or not w[:1].isalpha() for w in text.split())


def _is_heading(line):
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    if BULLET_RE.match(stripped):
        return False
    words = len(stripped.split())
    if stripped.endswith(":") and words <= 10:
        return True
    if words > 6 or stripped.endswith((".", ";", ",")):
        return False
    lowered = stripped.lower()
    return (stripped.isupper() or lowered.startswith(DROP_STARTS)
            or (_title_like(stripped) and any(k in lowered for k in KEEP_HEADS + DROP_HEADS)))


def extract_requirements(job_description, limit=40):
    """Pull requirement lines out of a pasted JD. Works with or without headings."""
    requirements, seen = [], set()
    mode = "bullets"  # bullets = only bulleted lines, keep = also plain lines, drop = ignore

    def add(item):
        item = item.strip(" \t;•")
        key = normalise_text(item)
        if len(item) >= 4 and key not in seen:
            seen.add(key)
            requirements.append(item)

    for raw in (job_description or "").splitlines():
        line = raw.strip()
        if not line or META_RE.match(line):
            continue
        bullet = BULLET_RE.match(line)
        if not bullet and _is_heading(line):
            head = re.sub(r"^#+\s*", "", line).rstrip(":").strip().lower()
            if any(k in head for k in DROP_HEADS):
                mode = "drop"
            elif any(k in head for k in KEEP_HEADS):
                mode = "keep"
            else:
                mode = "keep"
            continue
        if mode == "drop":
            continue
        if bullet:
            add(bullet.group(1))
        elif mode == "keep" and 3 <= len(line.split()) <= 40:
            add(line)

    if not requirements:  # JD is just paragraphs: keep sentences that sound like requirements
        sentences = re.split(r"(?<=[.!?])\s+|\n+", job_description or "")
        for sentence in sentences:
            if any(cue in sentence.lower() for cue in CUE_WORDS) and 3 <= len(sentence.split()) <= 40:
                add(sentence)
    return requirements[:limit]


# ----------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------
JOINERS = {"and", "or", "of", "to", "in", "for", "with", "the", "a", "an", "on", "by", "as", "&"}


def join_wrapped_lines(resume_text):
    """PDF text wraps mid-sentence. Re-join a line to the previous one when it clearly continues it."""
    merged = []
    for raw in (resume_text or "").splitlines():
        line = raw.strip()
        if not line:
            merged.append("")
            continue
        starts_bullet = re.match(r"^([•▪◦●■‣\-–*]|\d{1,2}[.)])\s*", line)
        prev = merged[-1] if merged else ""
        continues = bool(prev) and not starts_bullet and (
            line[:1].islower() or prev.endswith((",", "-", "–")) or prev.split()[-1].lower() in JOINERS)
        if continues:
            merged[-1] = prev + " " + line
        else:
            merged.append(line)
    return "\n".join(merged)


def clip(text, limit=300):
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def split_resume_sentences(resume_text):
    resume_text = join_wrapped_lines(resume_text)
    parts = re.split(r"\n+|\s*[|•]\s*|(?<=[.!?])\s+(?=[A-Z0-9\"'])", resume_text or "")
    return [p.strip(" \t•-\u2022") for p in parts if len(p.strip()) >= 12]


LABEL_RE = re.compile(r"^\s*(?:education|experience|qualifications?|skills?|requirements?)\s*:\s*", re.I)


def clean_label(requirement):
    """Drop a leading label such as 'Education:' so it is not matched against resume headings."""
    return LABEL_RE.sub("", requirement)


# In "Bachelor's degree in a quantitative field" these words describe the field loosely. The resume names the
# actual subject (Statistics), so requiring the word itself caused false gaps.
DEGREE_FILLER = {"quantitative", "technical", "related", "similar", "equivalent", "discipline", "disciplines",
                 "subject", "subjects", "area", "areas", "relevant", "analytical", "scientific"}


def key_terms(requirement, limit=8):
    """Meaningful terms in a requirement as a list of {stem, display, weight}."""
    found = {}
    is_degree = any(stem(t.lower()).startswith("edu_") for t in tokenize(requirement))
    for idx, tok in enumerate(tokenize(clean_label(requirement))):
        low = tok.lower()
        if is_degree and low in DEGREE_FILLER:
            continue
        if low in STOPWORDS or re.fullmatch(r"\d+\+?", low):
            continue
        if len(low) < 2 and low not in {"r", "c"}:
            continue
        st = stem(low)
        if st in STOPWORDS:
            continue
        weight = 1.0
        if (tok.isupper() and len(tok) >= 2) or any(ch in tok for ch in "+#.") or any(ch.isdigit() for ch in tok):
            weight = 2.0  # acronyms and technical names (SQL, C++, Node.js, ISO27001)
        elif tok[0].isupper() and idx > 0:
            weight = 1.5  # capitalised mid-sentence: probably a tool or proper noun
        if _raw_stem(low) in ACTION_STEMS:
            weight = 0.5
        if st not in found or weight > found[st]["weight"]:
            found[st] = {"stem": st, "display": tok, "weight": weight}
    ranked = sorted(found.values(), key=lambda t: (-t["weight"], -len(t["stem"])))
    return ranked[:limit]


CHUNK_RE = re.compile(r"[,;:()\[\]/]|\s[-–—]\s|\.\s")


def stem_sequence(text):
    """Ordered stems of meaningful words. None marks a punctuation break so phrases cannot span commas."""
    seq = []
    for chunk in CHUNK_RE.split(text or ""):
        seq.extend(stem(t.lower()) for t in tokenize(chunk) if t.lower() not in STOPWORDS)
        seq.append(None)
    return seq


def phrase_pairs(requirement, terms):
    """Adjacent key-term pairs inside the requirement, e.g. (fixed, income) or (asset, management)."""
    wanted = {t["stem"] for t in terms}
    pairs = set()
    for chunk in CHUNK_RE.split(clean_label(requirement)):
        prev, prev_action = None, False
        for tok in tokenize(chunk):
            low = tok.lower()
            current = None
            if not (low in STOPWORDS or re.fullmatch(r"\d+\+?", low) or (len(low) < 2 and low not in {"r", "c"})):
                candidate = stem(low)
                if candidate not in STOPWORDS:
                    current = candidate
            is_action = _raw_stem(low) in ACTION_STEMS
            if (current and prev and current != prev and current in wanted and prev in wanted
                    and not is_action and not prev_action):
                pairs.add((prev, current))
            prev, prev_action = current, is_action
    return sorted(pairs)


def has_pair(seq, first, second):
    return any(a == first and b == second for a, b in zip(seq, seq[1:]))


def weighted_coverage(terms, pairs, stems, seq):
    """Share of the requirement found. Each exact phrase ('fixed income') is a half-weight extra item, so scattered
    single words ('income statement') score lower than the real phrase."""
    total = sum(t["weight"] for t in terms) + 0.5 * len(pairs)
    if not total:
        return 0.0, [], []
    confirmed_pairs = [pair for pair in pairs if has_pair(seq, *pair)]
    in_pair = {s for pair in pairs for s in pair}
    confirmed = {s for pair in confirmed_pairs for s in pair}
    credit = 0.5 * len(confirmed_pairs)
    solid, loose = [], []
    for t in terms:
        if t["stem"] not in stems:
            continue
        credit += t["weight"]
        (loose if t["stem"] in in_pair and t["stem"] not in confirmed else solid).append(t)
    return credit / total, solid, loose


YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:(?:-|–|—|to)\s*(\d{1,2})\s*)?\+?\s*years?", re.I)
MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
RANGE_RE = re.compile(
    rf"(?:(?P<m1>{MONTHS})[a-z]*\.?,?\s+)?(?P<y1>(?:19|20)\d{{2}})\s*(?:-|–|—|to|until)\s*"
    rf"(?:(?P<m2>{MONTHS})[a-z]*\.?,?\s+)?(?P<y2>(?:19|20)\d{{2}}|present|current|now|ongoing|date|today)",
    re.I,
)
EDU_LINE_RE = re.compile(
    r"university|college|school|bachelor|master|degree|diploma|mba|bsc|msc|phd|gpa|cgpa|certificat", re.I)


def required_years(requirement):
    match = YEARS_RE.search(requirement)
    return int(match.group(1)) if match else None


def _month_index(abbr):
    return MONTHS.split("|").index(abbr.lower()[:4] if abbr.lower().startswith("sept") else abbr.lower()[:3]) \
        if abbr else None


def estimate_experience_years(resume_text):
    """Rough total of dated work ranges (overlaps merged, education lines skipped). None if no dates."""
    today = datetime.date.today()
    now = today.year + (today.month - 1) / 12
    spans = []
    for line in (resume_text or "").splitlines():
        if EDU_LINE_RE.search(line):
            continue
        for m in RANGE_RE.finditer(line):
            start = int(m.group("y1"))
            if m.group("m1"):
                start += _month_index(m.group("m1")) / 12
            end_raw = m.group("y2").lower()
            if end_raw.isdigit():
                end = int(end_raw) + (_month_index(m.group("m2")) / 12 if m.group("m2") else 0)
            else:
                end = now
            if start < end <= now + 1:
                spans.append((start, end))
    if not spans:
        return None
    spans.sort()
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return round(sum(e - s for s, e in merged), 1)


def analyse_resume(requirements, resume_text, strictness="Balanced"):
    clear_best, clear_cov, partial_cov = STRICTNESS[strictness]
    sentences = split_resume_sentences(resume_text)
    sentence_data = [(s, stem_set(s), stem_sequence(s)) for s in sentences]
    resume_stems = stem_set(resume_text)
    resume_seq = stem_sequence(resume_text)
    resume_years = estimate_experience_years(resume_text)
    results = []

    for index, requirement in enumerate(requirements, start=1):
        terms = key_terms(requirement)
        pairs = phrase_pairs(requirement, terms)
        years_needed = required_years(requirement)
        coverage, solid, loose = weighted_coverage(terms, pairs, resume_stems, resume_seq)
        missing = [t for t in terms if t["stem"] not in resume_stems]

        best_score, best_sentence = 0.0, None
        for sentence, stems, seq in sentence_data:
            if not terms:
                break
            score = weighted_coverage(terms, pairs, stems, seq)[0]
            if score > best_score or (score == best_score and best_sentence and len(sentence) < len(best_sentence)):
                best_score, best_sentence = score, sentence

        note = ""
        if not terms:
            if years_needed and resume_years is not None:
                status = CLEAR if resume_years + 0.5 >= years_needed else PARTIAL
                note = f"Estimated {resume_years} years of dated roles in the resume versus {years_needed} required."
            else:
                status = MANUAL
                note = "This requirement has no specific keywords to match, so please check it manually."
        elif best_score >= clear_best or (coverage >= clear_cov and best_score >= clear_best * 0.65):
            status = CLEAR
        elif coverage >= partial_cov:
            status = PARTIAL
        else:
            status = NONE

        if terms and years_needed and status in (CLEAR, PARTIAL):
            if resume_years is None:
                status = MANUAL
                note = f"The requirement asks for {years_needed}+ years, but dates could not be read from the resume."
            else:
                if resume_years + 0.5 < years_needed:
                    status = PARTIAL
                note = (f"Estimated {resume_years} years of dated roles versus {years_needed} required "
                        "(rough estimate, all roles combined).")

        found_terms = [t["display"] for t in solid + loose]
        if status == NONE:
            evidence = "No sentence in the resume covers this requirement."
            if found_terms:
                evidence += f" Only these scattered terms appear: {', '.join(found_terms)}."
            explanation = ("This does not prove the skill is missing. The resume text did not contain the key terms "
                           "together. If you have this experience, add it with specific, accurate wording.")
        else:
            evidence = clip(best_sentence) if best_sentence else "Related terms appear, but not together in one sentence."
            if status == CLEAR:
                explanation = "Most key terms appear together in the resume. Check the quotation to confirm the depth."
            elif status == MANUAL:
                explanation = note
            else:
                explanation = "Some key terms appear, but not all parts of the requirement are shown. Confirm manually."
        if note and status != MANUAL:
            explanation += " " + note

        results.append({
            "id": index, "requirement": requirement, "status": status,
            "evidence": evidence, "explanation": explanation,
            "matched": [t["display"] for t in solid], "loose": [t["display"] for t in loose],
            "missing": [t["display"] for t in missing], "terms": [t["display"] for t in terms],
        })
    return results


# ----------------------------------------------------------------------------
# Advice and free resources (category hints are for advice only, never for matching)
# ----------------------------------------------------------------------------
CATEGORIES = [
    {"id": "investment", "keywords": ["investment", "equity", "economic", "economics", "macroeconomic", "monetary", "asset",
                                      "portfolio", "bond", "valuation", "cfa", "fixed", "income", "fund", "securities",
                                      "broker", "forecasting", "financial", "finance"],
     "advice": "Write a one to two page research note on a listed company or asset class using only public data: the thesis, key figures, valuation view, main risks and a clear recommendation. Add a short plain-language update on one macro event (inflation, interest rates or a policy decision).",
     "resources": [
         {"title": "Investopedia: investing and economics guides", "url": "https://www.investopedia.com/", "note": "Free explainers; check the date of each article."},
         {"title": "St. Louis Fed: education resources", "url": "https://www.stlouisfed.org/education", "note": "Free lessons on economics and monetary policy."},
         {"title": "World Bank Open Learning Campus", "url": "https://olc.worldbank.org/", "note": "Free courses on economics and finance."},
         {"title": "CFA Institute", "url": "https://www.cfainstitute.org/", "note": "Official CFA program information; some resources are free."},
     ]},
    {"id": "software", "keywords": ["python", "java", "javascript", "typescript", "c++", "c#", "html", "css", "react",
                                      "angular", "vue", "node.js", "api", "git", "programming", "coding", "software",
                                      "backend", "frontend", "django", "flask", "docker", "kubernetes", "devops",
                                      "linux", "aws", "azure", "cloud", "algorithm", "testing", "develop"],
     "advice": "Build a small project that uses this skill, push it to a public repository with a clear README, and describe what it does in one resume bullet.",
     "resources": [
         {"title": "freeCodeCamp: free coding curriculum", "url": "https://www.freecodecamp.org/learn", "note": "Free, project-based."},
         {"title": "Harvard CS50 (free to audit)", "url": "https://cs50.harvard.edu/x/", "note": "Strong foundations in programming."},
         {"title": "MDN Web Docs: Learn web development", "url": "https://developer.mozilla.org/en-US/docs/Learn_web_development", "note": "Free web development guides."},
     ]},
    {"id": "data", "keywords": ["sql", "excel", "tableau", "power", "bi", "statistics", "spreadsheet", "pandas", "numpy",
                                  "visualization", "etl", "pivot", "regression", "dataset", "data"],
     "advice": "Pick a public dataset, answer three business questions with it, and save the result as a short report or dashboard with your method written down.",
     "resources": [
         {"title": "Kaggle Learn: free micro-courses", "url": "https://www.kaggle.com/learn", "note": "Python, SQL, data visualization."},
         {"title": "SQLBolt: interactive SQL lessons", "url": "https://sqlbolt.com/", "note": "Free hands-on SQL."},
         {"title": "Microsoft Support: Excel help and training", "url": "https://support.microsoft.com/en-us/excel", "note": "Official Excel guides."},
     ]},
    {"id": "office", "keywords": ["word", "powerpoint", "outlook", "office", "docs", "slides", "presentation"],
     "advice": "Produce one polished document and one short slide deck on a practice topic, focusing on clear headings, tables and consistent formatting.",
     "resources": [
         {"title": "Microsoft Learn: free training catalogue", "url": "https://learn.microsoft.com/en-us/training/", "note": "Search for Microsoft 365 topics."},
         {"title": "Microsoft Support: Office help", "url": "https://support.microsoft.com/en-us/office", "note": "Official how-to guides."},
     ]},
    {"id": "communication", "keywords": ["communication", "writing", "written", "verbal", "interpersonal", "negotiat",
                                          "speaking", "stakeholder", "presentation", "email", "grp_communication"],
     "advice": "Draft a clear one-page briefing or email on a work topic, then ask someone to review it for tone and clarity. Practise explaining it aloud in two minutes.",
     "resources": [
         {"title": "OpenLearn: Effective communication in the workplace", "url": "https://www.open.edu/openlearn/money-business/effective-communication-the-workplace/", "note": "Free Open University course."},
         {"title": "Purdue OWL: writing lab", "url": "https://owl.purdue.edu/", "note": "Free guides on report and business writing."},
     ]},
    {"id": "management", "keywords": ["project", "agile", "scrum", "planning", "budget", "mentor", "prioritiz", "deadline",
                                       "roadmap", "grp_manage", "grp_lead", "kanban", "delivery"],
     "advice": "Plan a small real or fictional project with scope, timeline, owners and risks. Track it for two weeks and write a short retrospective on what you would change.",
     "resources": [
         {"title": "Atlassian Agile Coach", "url": "https://www.atlassian.com/agile", "note": "Free guides to agile and project management."},
         {"title": "The Scrum Guide", "url": "https://scrumguides.org/", "note": "Free official guide."},
     ]},
    {"id": "marketing", "keywords": ["seo", "grp_marketing", "social", "media", "content", "campaign", "brand", "copywriting",
                                      "advertising", "ads", "crm", "email", "growth"],
     "advice": "Plan a mock campaign with a goal, audience, channels, a content calendar and the metrics you would track. Write a short results review using sample numbers.",
     "resources": [
         {"title": "HubSpot Academy: free courses", "url": "https://academy.hubspot.com/", "note": "Free marketing, sales and CRM courses."},
         {"title": "Google Skillshop", "url": "https://skillshop.withgoogle.com/", "note": "Free Google Ads and Analytics training."},
     ]},
    {"id": "finance", "keywords": ["accounting", "audit", "tax", "payroll", "bookkeeping",
                                    "reconciliation", "forecasting", "ifrs", "gaap", "invoice", "ledger"],
     "advice": "Use a fictional company to prepare a simple ledger, a bank reconciliation and a monthly summary. Check that totals reconcile and note each check you performed.",
     "resources": [
         {"title": "Khan Academy: Finance and capital markets", "url": "https://www.khanacademy.org/economics-finance-domain", "note": "Free lessons."},
         {"title": "OpenLearn: free business and finance courses", "url": "https://www.open.edu/openlearn/money-management", "note": "Free self-paced learning."},
     ]},
    {"id": "design", "keywords": ["design", "ui", "ux", "figma", "photoshop", "illustrator", "wireframe", "prototype",
                                   "creative", "branding", "typography"],
     "advice": "Redesign one screen or poster, show a before and after, and write three sentences explaining your design decisions. Add it to a simple portfolio page.",
     "resources": [
         {"title": "Google Design Resources", "url": "https://design.google/library/", "note": "Free articles and guidelines."},
         {"title": "OpenLearn: free design courses", "url": "https://www.open.edu/openlearn/", "note": "Search for design topics."},
     ]},
    {"id": "people", "keywords": ["recruit", "onboarding", "employee", "hr", "talent", "benefit", "hiring", "interview",
                                   "attendance", "hris", "hrms"],
     "advice": "Create a fictional process pack: a checklist, a tracker and one sample communication. Never practise with real people's data.",
     "resources": [
         {"title": "OpenLearn: free workplace skills courses", "url": "https://www.open.edu/openlearn/work-skills", "note": "Free workplace learning."},
         {"title": "CIPD: knowledge and guides", "url": "https://www.cipd.org/en/knowledge/", "note": "Public guides; some content may need membership."},
     ]},
    {"id": "customer", "keywords": ["sales", "grp_customer", "crm", "account", "service", "retention", "negotiation", "support"],
     "advice": "Write two realistic customer scenarios and a response for each. Add what you would measure (response time, satisfaction) and how you would follow up.",
     "resources": [
         {"title": "HubSpot Academy: free sales and service courses", "url": "https://academy.hubspot.com/", "note": "Free certifications."},
     ]},
    {"id": "soft", "keywords": ["grp_problem", "grp_collab", "critical", "thinking", "detail", "accuracy", "adaptab",
                                 "organiz", "initiative", "creativity", "grp_analysis"],
     "advice": "Prepare two short STAR stories (Situation, Task, Action, Result) showing this skill. Only use real outcomes.",
     "resources": [
         {"title": "OpenLearn: free workplace skills courses", "url": "https://www.open.edu/openlearn/work-skills", "note": "Free self-paced options."},
     ]},
    {"id": "education", "keywords": ["edu_bachelor", "edu_master", "edu_phd", "diploma", "certif", "qualification",
                                      "licen", "accredit"],
     "advice": "Check the education section states the qualification, institution and dates. If you are missing a certificate, look for an accredited free-to-audit course and list it as 'in progress' only if that is true.",
     "resources": [
         {"title": "edX: courses (free to audit)", "url": "https://www.edx.org/", "note": "University-level courses."},
         {"title": "Coursera: courses (many free to audit)", "url": "https://www.coursera.org/", "note": "Check which parts are free."},
     ]},
]
CATEGORIES.insert(0, {"id": "statistics", "keywords": ["statistic", "statistical", "statistics", "regression", "hypothesis", "probability", "inference"],
     "advice": "Work one example end to end: pick a public dataset, state a hypothesis, run a test or a simple regression, and write what the result does and does not show.",
     "resources": [
         {"title": "Khan Academy: Statistics and probability", "url": "https://www.khanacademy.org/math/statistics-probability", "note": "Free lessons."},
         {"title": "Kaggle Learn: free micro-courses", "url": "https://www.kaggle.com/learn", "note": "Data and machine learning basics."},
     ]})
for _cat in CATEGORIES:
    _cat["stems"] = {stem(k) if not k.startswith(("grp_", "edu_")) else k for k in _cat["keywords"]}

DEFAULT_ADVICE = ("Turn this requirement into a checklist. Identify what you need to know, practise with a realistic "
                  "example, and save a work sample or note that shows what you learned.")
DEFAULT_RESOURCES = [
    {"title": "OpenLearn: free courses", "url": "https://www.open.edu/openlearn/", "note": "Free self-paced learning."},
    {"title": "Coursera (many courses free to audit)", "url": "https://www.coursera.org/", "note": "Check which parts are free."},
    {"title": "Khan Academy", "url": "https://www.khanacademy.org/", "note": "Free lessons in many subjects."},
]


def search_links(terms):
    query = " ".join(terms[:3]).strip()
    if not query:
        return []
    return [
        {"title": f"YouTube search: {query} tutorial", "url": "https://www.youtube.com/results?search_query=" + quote_plus(query + " tutorial"), "note": "Free videos; check the channel quality."},
        {"title": f"Coursera search: {query}", "url": "https://www.coursera.org/search?query=" + quote_plus(query), "note": "Many courses can be audited for free."},
        {"title": f"Web search: free {query} course", "url": "https://duckduckgo.com/?q=" + quote_plus("free " + query + " course"), "note": "General search for free courses."},
    ]


def search_query_terms(requirement, terms):
    """Words for a search link: real phrases first (fixed income), else the strongest nouns."""
    display = {t["stem"]: t["display"] for t in terms}
    words = []
    for first, second in phrase_pairs(requirement, terms)[:2]:
        for item in (first, second):
            if display[item] not in words:
                words.append(display[item])
    if not words:
        words = [t["display"] for t in terms if t["weight"] > 0.5][:3]
    return words[:4]


def improvement_advice(requirement):
    """Return (advice text, resource list) for any requirement."""
    terms = key_terms(requirement)
    stems = {t["stem"] for t in terms}
    scored = sorted(((len(stems & cat["stems"]), cat) for cat in CATEGORIES), key=lambda x: -x[0])
    top = scored[0][0] if scored else 0
    chosen = [cat for hits, cat in scored if top and hits == top][:2]
    if len(chosen) > 1 and top < 2:
        chosen = chosen[:1]  # a one-word tie is too weak to add a second field's links  # only the best-fitting field(s)
    advice = chosen[0]["advice"] if chosen else DEFAULT_ADVICE
    resources, seen = [], set()
    for cat in chosen:
        for res in cat["resources"]:
            if res["url"] not in seen:
                resources.append(res)
                seen.add(res["url"])
    resources = resources[:3] or DEFAULT_RESOURCES[:2]
    for res in search_links(search_query_terms(requirement, terms))[:2]:
        if res["url"] not in seen:
            resources.append(res)
            seen.add(res["url"])
    return advice, resources


# ----------------------------------------------------------------------------
# Local AI helpers
# ----------------------------------------------------------------------------
def call_local_ai(messages, num_predict=550, model=None):
    import ollama
    response = ollama.chat(model=model or MODEL_NAME, messages=messages,
                           options={"temperature": 0.2, "num_predict": num_predict})
    return response["message"]["content"]


def strip_unknown_links(text, allowed_urls):
    """Remove any URL the model invented that is not in the supplied list."""
    allowed = {u.rstrip("/") for u in allowed_urls}

    def check(match):
        url = match.group(0).rstrip(").,;]")
        return match.group(0) if url.rstrip("/") in allowed else ""
    text = re.sub(r"https?://[^\s)\]]+", check, text)
    text = re.sub(r"\(\s*\)|\[\s*\]", "", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def top_gaps(results, limit=5):
    order = {NONE: 0, PARTIAL: 1, MANUAL: 2}
    gaps = [r for r in results if r["status"] != CLEAR]
    return sorted(gaps, key=lambda r: order.get(r["status"], 3))[:limit]


# ----------------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------------
def main():
    import streamlit as st

    st.set_page_config(page_title="AI Resume & Skill Development Assistant", layout="wide")
    st.title("AI Resume & Skill Development Assistant")
    st.write(
        "Compare a resume with any job description, spot missing or underdeveloped skills, and get a personalised "
        "learning plan with free resources. Fast text matching does the comparison; a local AI model (optional) "
        "writes the four-week plan."
    )
    st.info(
        "This is a learning and screening aid, not a hiring decision. A missing keyword does not prove a person "
        "lacks a skill, so check the evidence manually. Links are curated free resources or search links; "
        "some third-party sites may have paid extras."
    )

    model = st.sidebar.text_input("Local AI model", MODEL_NAME,
                                  help="Larger models (for example llama3.1:8b or qwen2.5:7b) write better plans if your PC can run them. Run 'ollama pull <name>' first.")
    strictness = st.sidebar.selectbox("Matching strictness", list(STRICTNESS), index=1,
                                      help="Lenient flags fewer gaps. Strict needs more of a requirement's terms in the resume.")

    st.header("1. Upload Your Resume")
    resume = st.file_uploader("Choose your resume", type=["pdf", "docx", "txt"])
    resume_text = ""
    if resume is not None:
        try:
            resume_text = extract_resume_text(resume)
            st.success(f"Resume uploaded: {resume.name}")
            with st.expander("Preview extracted resume text"):
                st.text_area("Extracted text", resume_text, height=220, key="resume_preview")
            if not resume_text.strip():
                st.warning("No readable text was found. A scanned PDF may need OCR.")
        except Exception as exc:
            st.error(f"Could not read the resume: {exc}")

    st.header("2. Enter the Job Description")
    job_description = st.text_area("Paste the job description here", height=220, key="jd_input",
                                   placeholder="Paste the responsibilities, required skills and qualifications...")

    detected = extract_requirements(job_description) if job_description.strip() else []
    with st.expander(f"Requirements detected from the job description ({len(detected)})"):
        for number, item in enumerate(detected, start=1):
            st.markdown(f"{number}. {item}")
        if not detected:
            st.write("Nothing detected yet.")
    extra_text = st.text_area("Missing a requirement? Add it here (one per line)", height=100, key="extra_reqs",
                              help="Use this if a line from the job description was not detected above.")
    fingerprint = hashlib.sha256((resume_text + "||" + job_description + "||" + extra_text + "||" + strictness).encode("utf-8", errors="ignore")).hexdigest()

    if st.button("Analyse Resume", type="primary"):
        if not resume:
            st.warning("Please upload a resume first.")
        elif not resume_text.strip():
            st.warning("Could not extract readable text from the resume.")
        elif not job_description.strip():
            st.warning("Please enter the job description.")
        else:
            requirements = list(detected)
            known = {normalise_text(r) for r in requirements}
            for line in extra_text.splitlines():
                if line.strip() and normalise_text(line) not in known:
                    requirements.append(line.strip())
                    known.add(normalise_text(line))
            if not requirements:
                st.warning("No requirements could be found. Try pasting bulleted responsibilities or qualifications.")
            else:
                with st.spinner("Matching requirements against the resume..."):
                    st.session_state["results"] = analyse_resume(requirements, resume_text, strictness)
                    st.session_state["fingerprint"] = fingerprint
                    st.session_state.pop("learning_plan", None)
                st.success(f"Compared {len(requirements)} requirements.")

    results = st.session_state.get("results")
    current = results is not None and st.session_state.get("fingerprint") == fingerprint

    if current:
        st.header("3. Resume Analysis")
        st.caption("Results come from keyword matching with conservative rules. Check the evidence manually.")
        cols = st.columns(4)
        for col, status in zip(cols, STATUSES):
            col.metric(status, sum(1 for r in results if r["status"] == status))

        for status in STATUSES:
            with st.expander(status, expanded=(status != CLEAR)):
                section = [r for r in results if r["status"] == status]
                if not section:
                    st.write("None.")
                for item in section:
                    st.markdown(f"**Requirement {item['id']}:** {item['requirement']}")
                    st.markdown(f"**Resume evidence:** {item['evidence']}")
                    if item["terms"]:
                        st.markdown(f"**Terms found:** {', '.join(item['matched']) or 'none'}  \n"
                                    f"**Found only as separate words (phrase not matched):** {', '.join(item['loose']) or 'none'}  \n"
                                    f"**Terms not found:** {', '.join(item['missing']) or 'none'}")
                    st.markdown(f"**Explanation:** {item['explanation']}")
                    st.divider()

        gaps = [r for r in results if r["status"] != CLEAR]
        st.header("4. Areas for Improvement")
        if not gaps:
            st.write("No gaps were flagged. Still read the evidence, since keyword matching cannot judge depth.")
        for item in gaps:
            advice, resources = improvement_advice(item["requirement"])
            with st.expander(f"{item['requirement']} — {item['status']}"):
                st.markdown(f"**What the resume shows:** {item['evidence']}")
                st.markdown(f"**How to improve:** {advice}")
                st.markdown("**Free learning resources:**")
                for res in resources:
                    st.markdown(f"- [{res['title']}]({res['url']}) — {res['note']}")
                st.markdown("**Proof of progress:** Save a practice exercise, work sample or an accurate resume bullet that shows the skill.")

        st.header("5. Personalised Four-Week Learning Plan")
        st.write("Optional. Your local AI model turns the top five gaps into weekly actions.")
        if st.button("Generate AI Learning Plan"):
            selected = top_gaps(results)
            blocks, plan_resources, seen = [], [], set()
            for g in selected:
                gap_res = improvement_advice(g["requirement"])[1][:4]
                listing = "\n".join(f"    * {r['title']}" for r in gap_res)
                blocks.append(f"- Gap: {g['requirement']} (status: {g['status']}; terms not found: "
                              f"{', '.join(g['missing']) or 'n/a'})\n  Resources for this gap:\n{listing}")
                for res in gap_res:
                    if res["url"] not in seen:
                        plan_resources.append(res)
                        seen.add(res["url"])
            gap_text = "\n".join(blocks) or "No gaps flagged; suggest ways to strengthen the listed requirements."
            try:
                with st.spinner("The local AI is writing a four-week plan..."):
                    plan = call_local_ai([
                        {"role": "system", "content": (
                            "You are a practical career-development coach. Write a concise four-week learning plan "
                            "using only the job requirements and gaps supplied. Describe gaps as missing or underdeveloped "
                            "resume evidence, never as proof the person lacks a skill. For each week give one goal, two "
                            "activities and one tangible output. Name a resource only by the exact title given under the "
                            "matching gap, without writing its URL (links are listed separately). Never invent course names, video titles or URLs; if no listed resource fits a "
                            "task, say 'practise with public data' instead. Keep it under 350 words with simple headings.")},
                        {"role": "user", "content": "Job description:\n" + job_description[:3500] +
                         "\n\nPriority gaps, each with its own resources:\n" + gap_text},
                    ], num_predict=800, model=model)
                st.session_state["learning_plan"] = strip_unknown_links(plan, [r["url"] for r in plan_resources])
                st.session_state["plan_resources"] = plan_resources
            except Exception as exc:
                st.error(f"Could not generate the plan: {exc}")
        if st.session_state.get("learning_plan"):
            st.markdown(st.session_state["learning_plan"])
            st.markdown("**Resources for this plan**")
            for res in st.session_state.get("plan_resources", []):
                st.markdown(f"- [{res['title']}]({res['url']}) — {res['note']}")
    elif results is not None:
        st.info("The resume, job description or strictness changed. Click Analyse Resume to refresh.")

    st.header("6. Optional: Extract Required Skills with AI")
    if st.button("Extract Required Skills with AI"):
        if not job_description.strip():
            st.warning("Please enter a job description first.")
        else:
            try:
                with st.spinner("Local AI is extracting key requirements..."):
                    skills = call_local_ai([
                        {"role": "system", "content": "Extract the most important technical, tool, domain and interpersonal requirements from the job description. Return a bullet list of at most 15 distinct items. Do not add anything not in the text."},
                        {"role": "user", "content": job_description[:4500]},
                    ], num_predict=300, model=model)
                st.markdown(skills)
            except Exception as exc:
                st.error(f"Could not extract skills: {exc}")

    st.header("7. Optional: Test Local AI Connection")
    if st.button("Test AI Connection"):
        try:
            reply = call_local_ai([{"role": "user", "content": "Reply with exactly: AI connection successful"}], num_predict=20, model=model)
            st.success(reply.strip())
        except Exception as exc:
            st.error(f"AI connection failed: {exc}")


if __name__ == "__main__":
    main()
