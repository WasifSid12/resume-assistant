import hashlib, json, os, sys, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(__file__))
from _engine import (extract_requirements, analyse_resume, improvement_advice,
                     top_gaps, strip_unknown_links, STRICTNESS)

KEY = os.environ.get("GEMINI_API_KEY", "")           # set in Vercel, never in code
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
DAILY_CAP = int(os.environ.get("DAILY_CAP", "80"))   # keep below the free-tier daily limit
_state, _cache = {"day": "", "n": 0}, {}             # best effort: resets when the server restarts

SYSTEM = ("You are a practical career coach. Write a concise four-week learning plan using only the gaps given. "
          "Describe gaps as missing or underdeveloped resume evidence, never proof the person lacks a skill. "
          "Per week: one goal, two activities, one tangible output. Never write URLs or invent course names. "
          "Under 300 words, simple headings.")


def ask_ai(gap_text, jd):
    if not KEY:
        raise RuntimeError("no API key set")
    today = time.strftime("%Y-%m-%d")
    if _state["day"] != today:
        _state.update(day=today, n=0)
    if _state["n"] >= DAILY_CAP:
        raise RuntimeError("daily AI cap reached")
    _state["n"] += 1
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": "Job description:\n" + jd[:2500] + "\n\nGaps:\n" + gap_text}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 700}}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "x-goog-api-key": KEY})
    for attempt in (0, 1):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                d = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt == 0:
                time.sleep(3)
                continue
            raise
    u = d.get("usageMetadata", {})
    return d["candidates"][0]["content"]["parts"][0]["text"], u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)


def template_plan(gaps):
    parts = []
    for week, g in enumerate(gaps[:4], 1):
        parts.append(f"**Week {week}: {g['requirement']}**\n\n{improvement_advice(g['requirement'])[0]}")
    return "\n\n".join(parts) or "No gaps flagged. Strengthen the evidence for each requirement."


def run(data):
    resume, jd = (data.get("resume") or "")[:12000], (data.get("jd") or "")[:6000]
    strict = data.get("strictness") if data.get("strictness") in STRICTNESS else "Balanced"
    if not resume.strip() or not jd.strip():
        raise ValueError("Please provide both a resume and a job description.")
    key = hashlib.sha256((resume + "|" + jd + "|" + strict).encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    reqs = extract_requirements(jd)
    if not reqs:
        raise ValueError("No requirements found. Paste bulleted responsibilities or qualifications.")
    results = analyse_resume(reqs, resume, strict)
    gaps, allowed, blocks = top_gaps(results), [], []
    for g in gaps:
        advice, res = improvement_advice(g["requirement"])
        g["advice"] = advice
        # only specific resources; search-result links are not verified free resources
        g["resources"] = [r for r in res if "search" not in r["title"].lower()]
        allowed += [r["url"] for r in g["resources"]]
        blocks.append(f"- {g['requirement']} (status: {g['status']}; terms not found: {', '.join(g['missing']) or 'n/a'})")
    plan, source, tin, tout = None, "ai", 0, 0
    if gaps:
        try:
            plan, tin, tout = ask_ai("\n".join(blocks), jd)
            plan = strip_unknown_links(plan, allowed)
        except Exception as exc:
            plan, source = template_plan(gaps), "template (AI unavailable: %s)" % str(exc)[:80]
    else:
        plan, source = template_plan(gaps), "template"
    out = {"results": results, "gaps": gaps, "plan": plan, "plan_source": source,
           "tokens": {"input": tin, "output": tout}}
    _cache[key] = out
    return out


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            out, code = run(json.loads(self.rfile.read(n) or b"{}")), 200
        except Exception as exc:
            out, code = {"error": str(exc)}, 400
        body = json.dumps(out, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
