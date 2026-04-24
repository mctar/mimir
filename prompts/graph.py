from __future__ import annotations


def mindmap_system(max_nodes: int, topic: str | None) -> str:
    topic_line = f'- Meeting context: "{topic}"' if topic else ""
    return f"""You generate mind-map graphs from meeting transcripts. Return ONLY valid JSON.

Rules:
- Max {max_nodes} nodes. Merge lesser concepts to stay under limit.
- Node: {{"id":"n1","label":"Short Name","group":"Category"}}
  - label: 2-4 words, title case
  - group: broad category for color clustering
- Edge: {{"source":"n1","target":"n2","label":"verb"}}
  - label: 1-2 word relationship (e.g. "drives", "enables", "part of")
- Preserve existing node IDs. Add only genuinely important new concepts.
- Remove nodes that are no longer relevant as the conversation evolves.
- Create edges that reveal the STRUCTURE of the discussion, not just proximity.
{topic_line}

Return: {{"nodes":[...],"edges":[...],"summary":"<2-3 sentence summary of all key points discussed so far>"}}"""


SMALL_MODEL_GRAPH_PREFIX = """CRITICAL: Output ONLY the raw JSON object. No thinking, no reasoning, no explanation, no markdown fences.

GRAPH EVOLUTION (follow strictly):
- You MUST add new nodes for every new concept, person, or topic in the NEW SEGMENT
- Always evolve the graph — never return it unchanged. The conversation is progressing, the graph must too.
- Every node MUST connect to at least 2 different nodes — no orphans
- NEVER create a star/hub where all nodes link to one central node
- Create cross-connections between related concepts, not just to the main topic
- Vary relationship labels: "enables", "requires", "part of", "contrasts", "drives", "informs", "blocks"
- Use "group" field (not "type") for node category

"""
