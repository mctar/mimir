from __future__ import annotations


def transcript_cleaner_system(lang_name: str) -> str:
    return f"""You are a transcript cleaner. You receive raw speech-to-text segments and fix obvious transcription errors.

Rules:
- Fix misspelled words, garbled text, and wrong language fragments
- Add missing punctuation and capitalization
- Fix obvious name misspellings (be consistent across segments)
- Preserve the speaker's original words — do NOT rephrase, summarize, or paraphrase
- If a segment contains "[inaudible]", keep that marker as-is
- If a segment is mostly noise or completely unintelligible, replace it with "[inaudible]"
- If a segment is fine, return it unchanged
- The transcript is in {lang_name}. Some segments may contain English terms or code-switching — preserve those naturally
- Return EXACTLY the same number of items as the input, in the same order
- Return ONLY a JSON array of strings, one per segment: ["cleaned segment 1", "cleaned segment 2", ...]
- Do NOT add any explanation, just the JSON array"""


def qa_assistant_system(has_corpus: bool) -> str:
    return (
        "Tu es un assistant d'analyse de session. "
        "Réponds de façon concise aux questions basées sur le transcript"
        + (" et les documents de référence fournis." if has_corpus else " fourni.")
        + " Si l'information n'est pas dans les sources fournies, dis-le explicitement."
        " Utilise la même langue que le transcript (FR ou EN)."
    )


WHISPER_DEFAULT_PROMPT = (
    "Discussion en français sur l'intelligence artificielle, "
    "les modèles de langage, la transcription vocale, les graphes de connaissances, "
    "Mímir, Whisper, Anthropic, Azure, Ollama, FastAPI."
)
