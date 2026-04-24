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


VOCABULARY_HINTS = (
    # Marques & entités
    "Capgemini, Capgemini Invent, WNS, Kipi.ai, McKinsey, HFS, Forrester, IDC, ISG, "
    "Microsoft, Google, AWS, Mistral AI, NVIDIA, Euronext, NYSE, "
    # Concepts stratégiques IO
    "Intelligent Operations, IOPS, Agentic AI, Agentic operations, BPO, BPS, "
    "Digital BPS, hyper-automation, Transform-then-Run, outcome-based, outcomes-based, "
    "FTE, CXO, GEB, CMD, Capital Markets Day, Comex, "
    # Deals & projets
    "Valor, Marvel, "
    # Personnes (participants Day 1 IO + corpus)
    "Aiman Ezzat, Franck Greverie, Fernando Alvarez, Oliver Pfeil, "
    "Keshav Murugesh, CP Duggal, Selva Vaidyanathan, "
    "Volker Darius, Kevin Campbell, Rob Walker, Kartik Ramakrishnan, "
    "Anirban Bose, Roshan Gya, "
    "Bhavesh, Jon Bell, Babu Mauze, "
    # Infra Mímir
    "Mímir, Ollama, FastAPI, DGX Spark, Parakeet, Canary, NeMo, Whisper, Anthropic, Gemini."
)

WHISPER_DEFAULT_PROMPT = (
    "Session de travail Capgemini Invent Day 1 sur l'Intelligent Operations (IOPS), "
    "avec le GEB et l'executive committee. Discussion stratégique sur le positionnement "
    "de Capgemini Invent, l'acquisition de WNS, les deals Valor et Marvel, "
    "l'IA agentique (Agentic AI), le modèle Transform-then-Run, la valeur outcome-based, "
    "et la proposition de valeur pour le Capital Markets Day."
)
