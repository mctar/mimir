import json, re
import pytest
import corpus


def test_synthesis_prompt_contains_transcript():
    prompt = corpus.build_synthesis_user_prompt("people discussed positioning", [])
    assert "people discussed positioning" in prompt


def test_synthesis_prompt_with_passages():
    passages = [{"title": "McKinsey Study", "content": "Market is large"}]
    prompt = corpus.build_synthesis_user_prompt("live content", passages)
    assert "McKinsey Study" in prompt
    assert "CORPUS CONTEXT" in prompt
    assert "do not override" in prompt


def test_qa_prompt():
    prompt = corpus.build_qa_user_prompt("transcript here", "Where is the disagreement?", [])
    assert "transcript here" in prompt
    assert "Where is the disagreement?" in prompt


def test_synthesis_system_prompt_has_json_format():
    assert '"themes"' in corpus.SYNTHESIS_SYSTEM_PROMPT
    assert "JSON only" in corpus.SYNTHESIS_SYSTEM_PROMPT


def test_parse_synthesis_json_valid():
    raw = '{"themes": [{"name": "Positioning", "alignment": ["a"], "disagreement": ["b"], "unresolved": ["c"]}]}'
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    assert match is not None
    data = json.loads(match.group())
    assert data["themes"][0]["name"] == "Positioning"


def test_parse_synthesis_json_with_preamble():
    raw = 'Here is the synthesis:\n{"themes": [{"name": "X", "alignment": [], "disagreement": [], "unresolved": []}]}'
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    assert match is not None
    data = json.loads(match.group())
    assert len(data["themes"]) == 1
