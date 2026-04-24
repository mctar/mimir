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


# ── cross_session_system ──────────────────────────────────────────────────────

def test_cross_session_insight_quality_rule():
    """Insights must name a consequence or decision implication, not just observe a connection."""
    prompt = cross_session_system("English", "en")
    assert "consequence" in prompt.lower() or "decision implication" in prompt.lower()
    assert "not merely observe" in prompt or "merely observe" in prompt


def test_cross_session_empty_array_preference():
    """Empty array preference must be explicitly stated for insights and tensions."""
    prompt = cross_session_system("English", "en")
    assert "empty" in prompt.lower()
    assert "omit" in prompt.lower()


# ── deck_spec_system ──────────────────────────────────────────────────────────

def test_deck_spec_is_in_english():
    """deck_spec_system must be written in English, not French."""
    prompt = deck_spec_system("CATALOG_PLACEHOLDER")
    # French markers that must be gone
    assert "Tu es" not in prompt
    assert "RÈGLES" not in prompt
    assert "CATALOGUE" not in prompt


def test_deck_spec_headline_quality_rule():
    """Slide headlines must be declarative assertions, not gerundive descriptions."""
    prompt = deck_spec_system("CATALOG_PLACEHOLDER")
    assert "declarative" in prompt.lower()
    assert "gerundive" in prompt.lower()


def test_deck_spec_language_instruction():
    """Output language must be derived from the recap language, not hardcoded."""
    prompt = deck_spec_system("CATALOG_PLACEHOLDER")
    assert "language of the" in prompt.lower()


# ── HTML_SLIDES_SYSTEM ────────────────────────────────────────────────────────

def test_html_slides_brand_color():
    """Background must use Capgemini Invent dark navy, not generic dark."""
    assert "#0C1A2E" in HTML_SLIDES_SYSTEM or "#0c1a2e" in HTML_SLIDES_SYSTEM


def test_html_slides_headline_quality_rule():
    """HTML slide prompt must enforce declarative headline rule."""
    assert "declarative" in HTML_SLIDES_SYSTEM.lower()
