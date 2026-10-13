"""Tests for the VibeFinder AI pipeline. A fake model stands in for Gemini, so no API key is needed."""

import contextlib
import io
import json

import pytest

from src.assistant import VibeFinder
from src.knowledge import detect_activity, load_notes, retrieve_notes
from src.llm import LLMError
from src.recommender import load_songs


class FakeLLM:
    """Returns scripted replies in order (or raises them if they are exceptions)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, json_mode=False, temperature=0.3):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(scope="module")
def data():
    with contextlib.redirect_stdout(io.StringIO()):
        songs = load_songs("data/songs.csv")
    return songs, load_notes("knowledge")


def make(data, replies=None):
    songs, notes = data
    return VibeFinder(songs, notes, FakeLLM(replies) if replies is not None else None)


# ---- retrieval -------------------------------------------------------------- #
def test_knowledge_notes_load_from_both_files(data):
    _, notes = data
    sources = {n["source"] for n in notes}
    assert sources == {"genres.md", "moods.md"}
    assert any(n["title"] == "lofi" for n in notes)


def test_retrieve_notes_uses_mood_genre_and_activity(data):
    _, notes = data
    found = [n["title"] for n in retrieve_notes(notes, {"genre": "lofi", "mood": "chill"}, "lofi for studying")]
    assert {"lofi", "chill", "studying"} <= set(found)
    assert detect_activity("music for the gym") == "workout"


def test_offline_keyword_parser(data):
    vf = make(data)
    assert vf.keyword_parse("songs for the gym")["mood"] == "intense"
    assert vf.keyword_parse("indie pop please")["genre"] == "indie pop"
    kpop = vf.keyword_parse("k-pop bangers")
    assert kpop["genre"] is None and kpop["unknown_genre"] == "k-pop"


def test_offline_run_retrieves_matching_songs(data):
    result = make(data).run("chill music for studying")
    assert result.source == "fallback"
    assert result.check_passed
    assert result.songs[0]["mood"] == "chill"
    assert result.notes, "should retrieve knowledge notes too"


# ---- RAG generation + guardrails --------------------------------------------- #
def test_gemini_answer_uses_retrieved_context(data):
    plan = json.dumps({"genre": None, "mood": "intense", "energy": 0.9})
    llm_answer = '"Storm Runner" by Voltline and "Gym Hero" by Max Pulse will keep you moving.'
    vf = make(data, [plan, llm_answer])
    result = vf.run("workout songs")
    assert result.source == "gemini"
    assert result.answer.startswith('"Storm Runner"')
    assert "SONGS:" in vf.llm.prompts[1] and "NOTES:" in vf.llm.prompts[1]  # retrieved data is in the prompt


def test_invented_song_triggers_retry_with_feedback(data):
    vf = make(data, ['{"mood": "intense", "energy": 0.9}',
                     'Try "Eye of the Tiger".',
                     'Try "Storm Runner" by Voltline.'])
    result = vf.run("gym songs")
    assert result.source == "gemini-retry"
    assert "Eye of the Tiger" in vf.llm.prompts[-1]
    assert result.check_passed


def test_two_bad_answers_fall_back_safely(data):
    vf = make(data, ['{"mood": "happy"}', '"Made Up Song"', '"Another Fake"'])
    result = vf.run("happy music")
    assert result.source == "fallback"
    assert result.check_passed
    assert "Made Up Song" not in result.answer


def test_gemini_outage_and_bad_json_never_crash(data):
    vf = make(data, ["this is not json", LLMError("quota exceeded")])
    result = vf.run("sad songs")
    assert result.source == "fallback"
    assert result.trace[0]["detail"]["method"] == "keyword parser (gemini failed)"


def test_invalid_plan_values_are_replaced(data):
    vf = make(data, ['{"genre": "polka", "mood": "spooky", "energy": 7}', '"Sunrise City" by Neon Echo'])
    result = vf.run("happy pop")
    assert result.prefs["genre"] == "pop"      # polka isn't in the catalog -> keyword parser value
    assert result.prefs["mood"] == "happy"
    assert result.prefs["energy"] == 1.0       # clamped to 0-1


def test_check_flags_catalog_song_that_was_not_retrieved(data):
    vf = make(data)
    songs = vf.run("chill study music").songs
    problems = vf.check('"Iron Tide" is perfect for this', songs)
    assert any("not retrieved" in p for p in problems)


# ---- input guardrails + confidence ------------------------------------------- #
def test_empty_input_is_rejected(data):
    result = make(data).run("   ")
    assert result.source == "error"
    assert "Please describe" in result.answer


def test_long_input_is_trimmed(data):
    result = make(data).run("chill " * 200)
    assert len(result.query) == 300
    assert result.trace[0]["step"] == "guardrail"


def test_missing_genre_lowers_confidence_and_says_so(data):
    vf = make(data)
    normal = vf.run("euphoric party music")
    kpop = vf.run("k-pop party music")
    assert kpop.confidence < normal.confidence
    assert "no k-pop songs" in kpop.answer


def test_quoted_phrases_are_not_mistaken_for_songs(data):
    vf = make(data)
    songs = vf.run("chill study music").songs
    answer = f'"{songs[0]["title"]}" by {songs[0]["artist"]} has that "late night" feeling.'
    assert vf.check(answer, songs) == []
    assert vf.check('"Fake Anthem" by Nobody', songs) != []
