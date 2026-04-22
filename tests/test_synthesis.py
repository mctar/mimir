import json, re


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
