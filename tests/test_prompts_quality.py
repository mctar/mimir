# tests/test_prompts_quality.py
from prompts.recap import BOARD_RECAP_SYSTEM, cross_session_system
from prompts.slides import deck_spec_system, HTML_SLIDES_SYSTEM


# ── BOARD_RECAP_SYSTEM ────────────────────────────────────────────────────────

def test_recap_has_four_question_filter():
    """The 4-question filter must be present as an explicit behavioral rule."""
    assert "What was decided or proposed" in BOARD_RECAP_SYSTEM
    assert "What is at stake if this is not acted on" in BOARD_RECAP_SYSTEM
    assert "What specifically differentiates this position" in BOARD_RECAP_SYSTEM
    assert "What is the recommended next action" in BOARD_RECAP_SYSTEM


def test_recap_has_contrast_example():
    """A before/after example must show the difference between generic and executive."""
    assert "REJECT" in BOARD_RECAP_SYSTEM
    assert "ACCEPT" in BOARD_RECAP_SYSTEM
    assert "Any firm could write this" in BOARD_RECAP_SYSTEM


def test_recap_has_self_review_step():
    """A mandatory self-review gate must precede the JSON output instruction."""
    assert "MANDATORY SELF-REVIEW" in BOARD_RECAP_SYSTEM
    assert "vague verbs" in BOARD_RECAP_SYSTEM


def test_recap_empty_list_rule():
    """Empty list preference must be stated explicitly."""
    assert "empty list is preferable" in BOARD_RECAP_SYSTEM.lower()
