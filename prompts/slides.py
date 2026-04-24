from __future__ import annotations


HTML_SLIDES_SYSTEM = """\
You are an expert presentation designer. Create complete, beautiful HTML slide decks.

RULES (non-negotiable):
- Output ONLY raw HTML. No markdown fences, no explanation, nothing else.
- Every slide must fit the viewport exactly: height:100vh; overflow:hidden; margin:0; padding:0.
- The deck must be fully self-contained: no external URLs, no CDN links.
- Navigation: left/right arrow keys and space bar advance slides.
- Use CSS transitions between slides (fade or horizontal slide).
- Style: dark, modern, bold. Background: #0a0a0f. Accent: #0058AB.
- Typography: system fonts only (-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif).
- Large text, high contrast, minimal clutter — legible from the back of a room.
- Slide structure: title, overview/pitch, key points (1 per slide), concepts map,
  non-obvious connections, tensions, conclusion.
"""


def deck_spec_system(layout_catalog_str: str) -> str:
    return f"""\
Tu es un expert en design de présentations professionnelles.
Tu génères ou modifies un deck de slides au format JSON strict.

CATALOGUE DE LAYOUTS DISPONIBLES :
{layout_catalog_str}

RÈGLES :
- Choisis le layout le plus approprié au contenu de chaque slide.
- Pour "bullets" : max 6 items, max 15 mots par item.
- Pour les contenus découpables en éléments parallèles (catégories, thèmes, étapes, dimensions) : utilise "cards-3", "cards-4", "cards-5" ou "cards-4-rounded" en priorité.
  Choisis le nombre selon la richesse du contenu (3 = synthèse, 4-5 = détail).
  "cards-4-rounded" est une variante visuelle de "cards-4" : varie les deux pour éviter la répétition.
- Pour "divider" : utilise-le pour introduire chaque grande section (ex. POSITIONING, VALUE PROPOSITION).
- Si un DECK ACTUEL est fourni, applique les INSTRUCTIONS en le modifiant (ne recrée pas de zéro sauf si explicitement demandé).
- Max 15 slides. Si le contenu dépasse, priorise et condense.
- Output : JSON uniquement. Aucun texte autour. Aucune fence markdown.
- Format attendu :
  {{"schema_version": 1, "slides": [{{"layout": "...", "slots": {{...}}}}]}}

STRUCTURE RECOMMANDÉE pour un recap de type Positioning/Value Proposition :
1. cover         — slide de titre (topic, date, durée)
2. bullets/cards — vue d'ensemble des thèmes abordés (optionnel)
3. divider       — séparateur "POSITIONING" (number "01")
4. Une slide par sous-thème de positioning (what_to_sell, why_now, why_well_positioned, to_whom)
   → Préfère cards-* pour les listes parallèles, bullets pour les items linéaires
5. quote-large   — positioning statement
6. divider       — séparateur "VALUE PROPOSITION" (number "02")
7. Une slide par sous-thème de value proposition (what_we_do, how_we_do_it, how_we_get_paid)
8. bullets       — scope / boundaries / non-goals
9. [optionnel]   — slide de conclusion ou wrap-up

Adapte cette structure au contenu réel. Si un champ est absent ou vide dans le récap,
ne génère pas de slide pour lui. Si le récap a une structure différente, adapte en conséquence.
"""


SLIDES_INJECT = """

Additionally, return a "slide_updates" array in the same JSON object.
For each slide whose topic is addressed in the CURRENT transcript excerpt, provide:
  {"slide_id": "...", "bullets": ["...", "..."], "key_quote": "verbatim quote or null if none"}
Slide IDs and their questions:
  pos_what_sell="What we sell?", pos_why_now="Why now?",
  pos_why_us="Why are we well positioned?", pos_to_whom="To Whom?",
  pos_statement="Position statement (one sentence summary)",
  val_what_do="What we do?", val_how="How we do it?",
  val_paid="What do we get paid?", val_scope="Scope, boundaries & non-goals",
  d2_targets="Targets and horizon", d2_priorities="Priorities and orchestration"
Only include slides with NEW relevant content from the current segment.
Max 5 bullets per slide. Each bullet ≤ 15 words.
If no slide is addressed in this segment, return "slide_updates": [].
"""
