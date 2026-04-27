from __future__ import annotations


GEB_ASE_IOPS_CONTEXT = (
    '"GEB ASE IOPS" stands for Group Executive Board – Accelerator Solution Environment – '
    "Intelligent Operations. This is a Capgemini executive programme at GEB level, "
    "focused on defining and activating the Intelligent Operations (IOPS) offer strategy."
)


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
        f"Contexte : {GEB_ASE_IOPS_CONTEXT} "
        "Réponds de façon concise aux questions basées sur le transcript"
        + (" et les documents de référence fournis." if has_corpus else " fourni.")
        + " Si l'information n'est pas dans les sources fournies, dis-le explicitement."
        " Utilise la même langue que le transcript (FR ou EN)."
    )


VOCABULARY_HINTS = (
    # Marques & entités
    "Capgemini, Capgemini Invent, WNS, Kipi.ai, McKinsey, HFS, Forrester, IDC, ISG, "
    "Microsoft, Google, AWS, Mistral AI, NVIDIA, Euronext, NYSE, "
    # Programme
    "GEB ASE IOPS, ASE, Accelerator Solution Environment, Group Executive Board, "
    # Concepts stratégiques IO
    "Intelligent Operations, IOPS, Agentic AI, Agentic operations, BPO, BPS, "
    "Digital BPS, hyper-automation, Transform-then-Run, outcome-based, outcomes-based, "
    "FTE, CXO, GEB, CMD, Capital Markets Day, Comex, "
    # Deals & projets
    "Valor, Marvel, "
    # Personnes (participants Day 1 IO + GEB + executive committee)
    # GEB
    "Aiman Ezzat, Fernando Alvarez, Nive Bhagat, Anirban Bose, "
    "Karine Brunet, Andrea Falleni, Cyril Garcia, Franck Greverie, "
    "Roshan Gya, Anne Lebel, Kartik Ramakrishnan, Michael Schulte, Jérôme Simeon, "
    # Extended exec committee
    "Pascal Brier, Kevin Campbell, Inma Casero, Sanjay Chalke, Volker Darius, "
    "Steffen Elsaesser, Patrick Ferraris, Stephen Hilton, Olivier Lepick, "
    "Ted Levine, Karine Marchat, Paul Margetts, Fabrice Mariaud, Rainer Mehl, "
    "Keshav Murugesh, Sarika Naik, Rajnish Nath, Niraj Parihar, Maria Pernas, "
    "Oliver Pfeil, Béatrice Speisser, Shin Tonomura, Volkmar Varnhagen, Jeroen Versteeg, "
    # Day 1 IO session additional (WNS executives, other participants)
    "CP Duggal, Selva Vaidyanathan, Rob Walker, "
    "Bhavesh, Jon Bell, Babu Mauze, "
    # Infra Mímir
    "Mímir, Ollama, FastAPI, DGX Spark, Parakeet, Canary, NeMo, Whisper, Anthropic, Gemini."
)

WHISPER_DEFAULT_PROMPT = (
    "GEB ASE IOPS (Group Executive Board – Accelerator Solution Environment – Intelligent Operations): "
    "Capgemini executive programme, Day 1 strategic working session on Intelligent Operations (IOPS). "
    "Agenda: align on positioning (what to sell, why now, why well-positioned, to whom) "
    "and value proposition (what we do, how we do it, how we get paid). "
    "Key topics: Transform-and-Run, Value game step-up, Asset/IP-led services, "
    "Front/Core/Back, market inflexion, transformation renewal, IOPs market momentum, "
    "agentic operations, Tri-pod (Transfo x Industry x Tech), orchestrator and neutral partner, "
    "CXO play, customer tier, client archetypes, value engine, E2E operational reinvention, "
    "static-to-dynamic services, evolution vs disruption, deals anatomy, "
    "capabilities aggregation, Value/Risks/Cash model, shared accountability. "
    "WNS acquisition context, deals Valor and Marvel, Capital Markets Day preparation."
)
