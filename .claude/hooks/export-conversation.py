#!/usr/bin/env python3
"""
Hook SessionEnd pour Claude Code.
Exporte automatiquement chaque conversation en markdown.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def extract_text_content(content):
    """Extrait le texte d'un contenu qui peut être string ou liste de blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_name = block.get("name", "unknown")
                    texts.append(f"[Outil: {tool_name}]")
                elif block.get("type") == "tool_result":
                    texts.append("[Résultat d'outil]")
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)
    return str(content)


def truncate_text(text, max_length=500):
    """Tronque le texte s'il est trop long."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def main():
    # Lire l'input JSON depuis stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Erreur JSON: {e}", file=sys.stderr)
        sys.exit(1)

    transcript_path = input_data.get("transcript_path")
    session_id = input_data.get("session_id", "unknown")
    cwd = input_data.get("cwd", ".")

    if not transcript_path or not Path(transcript_path).exists():
        print(f"Fichier transcript non trouvé: {transcript_path}", file=sys.stderr)
        sys.exit(0)

    # Créer le dossier de logs
    export_dir = Path(cwd) / ".claude" / "conversation-logs"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Lire le fichier JSONL de conversation
    messages = []
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        msg = json.loads(line)
                        messages.append(msg)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Erreur lecture transcript: {e}", file=sys.stderr)
        sys.exit(1)

    if not messages:
        print("Aucun message dans la conversation", file=sys.stderr)
        sys.exit(0)

    # Générer le markdown
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    filename = f"{date_str}_{session_id[:8]}.md"

    output_lines = [
        f"# Session Claude - {date_str}",
        f"",
        f"- **Date**: {date_str} {time_str}",
        f"- **Session ID**: {session_id}",
        f"- **Messages**: {len(messages)}",
        f"",
        f"---",
        f"",
    ]

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "user":
            role_label = "**Utilisateur**"
        elif role == "assistant":
            role_label = "**Claude**"
        else:
            role_label = f"**{role}**"

        text = extract_text_content(content)
        if text.strip():
            # Tronquer les messages très longs pour lisibilité
            text_display = truncate_text(text.strip(), max_length=1000)
            output_lines.append(f"### {role_label}")
            output_lines.append(f"")
            output_lines.append(text_display)
            output_lines.append(f"")
            output_lines.append(f"---")
            output_lines.append(f"")

    # Écrire le fichier markdown
    export_path = export_dir / filename

    # Si le fichier existe déjà (plusieurs sessions le même jour), ajouter un suffixe
    counter = 1
    while export_path.exists():
        filename = f"{date_str}_{session_id[:8]}_{counter}.md"
        export_path = export_dir / filename
        counter += 1

    try:
        with open(export_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"Conversation exportée: {export_path}")
    except Exception as e:
        print(f"Erreur écriture: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
