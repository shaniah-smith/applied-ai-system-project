"""Second retrieval source: short notes about genres, moods, and activities stored in knowledge/*.md."""

import re
from pathlib import Path
from typing import Dict, List, Optional

ACTIVITY_WORDS = {
    "studying": ["study", "studying", "homework", "focus", "concentrate", "work", "reading", "exam"],
    "workout": ["workout", "gym", "run", "running", "lift", "lifting", "exercise", "training", "cardio"],
    "sleep": ["sleep", "bed", "bedtime", "nap", "insomnia"],
}


def load_notes(folder: str = "knowledge") -> List[Dict]:
    """Split every markdown file into notes, one per '## heading'."""
    notes = []
    for path in sorted(Path(folder).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for block in re.split(r"^## ", text, flags=re.M)[1:]:
            title, _, body = block.partition("\n")
            notes.append({"title": title.strip().lower(), "text": body.strip(), "source": path.name})
    return notes


def detect_activity(query: str) -> Optional[str]:
    """Return 'studying', 'workout', or 'sleep' if the request mentions one."""
    words = set(re.findall(r"[a-z]+", query.lower()))
    for activity, keywords in ACTIVITY_WORDS.items():
        if words & set(keywords):
            return activity
    return None


def retrieve_notes(notes: List[Dict], prefs: Dict, query: str, k: int = 3) -> List[Dict]:
    """Rank notes by how well they match the parsed genre/mood and the activity in the query."""
    activity = detect_activity(query)
    query_words = set(re.findall(r"[a-z&]+", query.lower()))
    scored = []
    for note in notes:
        score = 0.0
        if note["title"] == str(prefs.get("genre", "")).lower():
            score += 3
        if note["title"] == str(prefs.get("mood", "")).lower():
            score += 3
        if activity and note["title"] == activity:
            score += 3
        score += 0.2 * len(query_words & set(re.findall(r"[a-z&]+", note["text"].lower())))
        if score >= 1:
            scored.append((score, note))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [note for _, note in scored[:k]]
