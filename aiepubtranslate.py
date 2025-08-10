#!/usr/bin/env python3
"""
Translate an EPUB while preserving HTML/CSS structure and repackage it as a valid EPUB.

Features:
- Optional context file (.txt) for translation notes/glossary.
- Automatic detection of poetry-like segments → use special poetic translation prompt.
- Default model: gpt-4o.

Usage:
  python translate_epub.py input.epub output.epub --target-lang de --source-lang en \
      --context-file context.txt --openai-api-key $OPENAI_API_KEY
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import tempfile
import time
import logging
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

from bs4 import BeautifulSoup, NavigableString, Tag
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

SKIP_TAGS = {'script', 'style', 'code', 'pre', 'svg', 'math'}
HTML_EXTS = {'.xhtml', '.html', '.htm'}
PLACEHOLDER_PREFIX = "__SEG__"

def is_html_like(path: Path) -> bool:
    return path.suffix.lower() in HTML_EXTS

def iter_text_nodes(soup: BeautifulSoup) -> Iterable[Tuple[NavigableString, Tag]]:
    for node in soup.find_all(string=True):
        parent = node.parent
        if not isinstance(parent, Tag):
            continue
        anc = parent
        skip = False
        while anc is not None:
            if isinstance(anc, Tag) and anc.name in SKIP_TAGS:
                skip = True
                break
            anc = anc.parent
        if skip:
            continue
        text = str(node)
        if not text or text.strip() == "":
            continue
        if not re.search(r"\w", text):
            continue
        yield node, parent

def chunk_segments(segments: List[str], max_chars: int = 8000) -> List[List[str]]:
    chunks, cur, cur_len = [], [], 0
    for seg in segments:
        l = len(seg)
        if cur and cur_len + l > max_chars:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(seg)
        cur_len += l
    if cur:
        chunks.append(cur)
    return chunks

def looks_like_poetry(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) >= 3:
        avg_len = sum(len(l) for l in lines) / len(lines)
        if avg_len < 60:
            return True
    return False

def build_translation_prompt(segments: List[str], source_lang: Optional[str], target_lang: str, context: Optional[str]) -> str:
    numbered = [f"{PLACEHOLDER_PREFIX}{i}: {s}" for i, s in enumerate(segments)]
    guidance = (
        "Du bist ein professioneller Übersetzer. Übersetze NUR die Inhalte nach den Doppelpunkten. "
        "Gib EXAKT so viele Übersetzungen zurück wie Eingabesegmente. "
        "Jedes Segment wird als EIN String ausgegeben, auch wenn es mehrere Zeilen enthält (Zeilenumbrüche beibehalten). "
        "Antwortformat: JSON-Liste der Übersetzungen in gleicher Reihenfolge, ohne zusätzliche Erklärungen."
        "Antworte ohne Code-Fences (keine ```), nur reines JSON."
    )
    if source_lang:
        guidance += f" Ausgangssprache: {source_lang}."
    guidance += f" Zielsprache: {target_lang}."
    if context:
        guidance += f"\n\nKONTEXT HINWEISE:\n{context}"
    return guidance + "\n\nSEGMENTE:\n" + "\n".join(numbered)

def build_poetry_prompt(segments: List[str], source_lang: Optional[str], target_lang: str, context: Optional[str]) -> str:
    numbered = [f"{PLACEHOLDER_PREFIX}{i}: {s}" for i, s in enumerate(segments)]
    guidance = (
        "Du bist ein literarischer Übersetzer für Gedichte. Übersetze NUR die Inhalte nach den Doppelpunkten so, "
        "dass sie poetisch klingen (Reim/Rhythmus wichtiger als Worttreue). "
        "Gib EXAKT so viele Übersetzungen zurück wie Eingabesegmente. "
        "Wenn ein Segment mehrere Zeilen hat, gib EINEN String mit Zeilenumbrüchen zurück. "
        "Antwortformat: JSON-Liste der Übersetzungen in gleicher Reihenfolge, ohne zusätzliche Erklärungen."
        "Antworte ohne Code-Fences (keine ```), nur reines JSON."
    )
    if source_lang:
        guidance += f" Ausgangssprache: {source_lang}."
    guidance += f" Zielsprache: {target_lang}."
    if context:
        guidance += f"\n\nKONTEXT HINWEISE:\n{context}"
    return guidance + "\n\nSEGMENTE:\n" + "\n".join(numbered)

def openai_translate(client: OpenAI, model: str, segments: List[str], source_lang: Optional[str], target_lang: str, context: Optional[str]) -> List[str]:
    """Translate segments with robust fallback across OpenAI SDK variants.
    Order of attempts per call:
    1) Responses API with JSON response_format (newer SDKs)
    2) Responses API without response_format (some SDKs)
    3) Chat Completions (legacy-compatible)
    """
    import json

    if not segments:
        return []

    def _call_openai(client: OpenAI, model: str, prompt: str, temperature: float = 0.4, max_output_tokens: int = 4096):
        # Try Responses with JSON formatting
        try:
            r = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            )
            return getattr(r, 'output_text', str(r)), 'responses_json'
        except TypeError:
            # Older SDK: no response_format arg
            pass
        except Exception:
            # Other runtime errors fall through to next attempt
            pass
        # Try Responses without response_format
        try:
            r = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            return getattr(r, 'output_text', str(r)), 'responses_text'
        except Exception:
            pass
        # Fallback to Chat Completions
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON for the user prompt."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        return chat.choices[0].message.content, 'chat'

    outputs: List[str] = []
    for chunk in chunk_segments(segments):
        # Choose prompt based on poetry detection
        poetic = all(looks_like_poetry(s) for s in chunk)
        logging.info('Chunk size: %d | Poetry mode: %s', len(chunk), 'yes' if poetic else 'no')
        if all(looks_like_poetry(s) for s in chunk):
            prompt = build_poetry_prompt(chunk, source_lang, target_lang, context)
        else:
            prompt = build_translation_prompt(chunk, source_lang, target_lang, context)

        content = None
        # Up to 3 retries with backoff
        for attempt in range(1, 4):
            try:
                content, mode = _call_openai(client, model, prompt, temperature=0.4, max_output_tokens=4096)
                logging.info('Model call mode: %s', mode)

                # 1) JSON parsen
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        trans = data
                    elif isinstance(data, dict):
                        for key in ("translations", "result", "items", "data"):
                            if key in data and isinstance(data[key], list):
                                trans = data[key]
                                break
                        else:
                            raise ValueError("JSON missing translations list")
                    else:
                        raise ValueError("Unexpected JSON root type")
                except Exception:
                    # 2) Fallback: pro Zeile – KEIN raise hier!
                    trans = [l.strip() for l in str(content).splitlines() if l.strip()]

                # 3) Reconciliation, falls Längen nicht passen
                if len(trans) != len(chunk):
                    logging.warning(
                        'Translation count mismatch: got %d, expected %d. Attempting reconciliation.',
                        len(trans), len(chunk)
                    )
                    if len(chunk) == 1 and len(trans) > 1:
                        # Ein Eingabesegment → mehrere Outputs: zusammenführen
                        trans = ["\n".join(map(str, trans))]
                    elif len(trans) > len(chunk) and len(trans) % len(chunk) == 0:
                        # Vielfaches: in Gruppen bündeln
                        group = len(trans) // len(chunk)
                        trans = [
                            "\n".join(map(str, trans[i*group:(i+1)*group]))
                            for i in range(len(chunk))
                        ]
                    else:
                        raw_preview = (str(content)[:300] + '…') if len(str(content)) > 300 else str(content)
                        logging.error("Raw model output preview: %s", raw_preview)
                        raise ValueError(f"Translation count mismatch: got {len(trans)}, expected {len(chunk)}")

                # 4) Jetzt IMMER anhängen – egal ob mismatch oder nicht
                outputs.extend([str(t) for t in trans])
                break  # success for diesen Chunk

            except Exception as e:
                logging.warning('Retry %d due to: %s', attempt, e)
                if attempt == 3:
                    raise
                time.sleep(2.0 * attempt)

    return outputs

def translate_html_file(path: Path, client: OpenAI, model: str,
                        source_lang: Optional[str], target_lang: str,
                        context: Optional[str]) -> None:
    html = path.read_text(encoding='utf-8', errors='ignore')
    is_xmlish = path.suffix.lower() in {'.xhtml', '.xml', '.opf', '.ncx'} or html.lstrip().startswith('<?xml')
    parser = 'xml' if is_xmlish else 'lxml'
    soup = BeautifulSoup(html, parser)
    nodes: List[NavigableString] = []
    segments: List[str] = []
    for node, _ in iter_text_nodes(soup):
        nodes.append(node)
        segments.append(str(node))

    if not segments:
        return

    translations = openai_translate(client, model, segments, source_lang, target_lang, context)

    for node, new_text in zip(nodes, translations):
        node.replace_with(NavigableString(new_text))

    path.write_text(str(soup), encoding='utf-8')

def update_opf_language(opf_path: Path, target_lang: str) -> None:
    if not opf_path.exists():
        return
    xml = opf_path.read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(xml, 'xml')
    dc_lang = soup.find(lambda tag: isinstance(tag, Tag) and tag.name.lower().endswith('language'))
    if dc_lang is None:
        metadata = soup.find(lambda tag: isinstance(tag, Tag) and tag.name.lower().endswith('metadata'))
        if metadata is None:
            return
        new_tag = soup.new_tag('dc:language')
        new_tag.string = target_lang
        metadata.append(new_tag)
    else:
        dc_lang.string = target_lang
    opf_path.write_text(str(soup), encoding='utf-8')

def safe_extract_epub(epub_path: Path, workdir: Path) -> Path:
    extract_dir = workdir / 'epub'
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(epub_path, 'r') as zf:
        zf.extractall(extract_dir)
    return extract_dir

def find_opf(root: Path) -> Optional[Path]:
    candidates = list(root.glob('**/*.opf'))
    return candidates[0] if candidates else None

def repackage_epub(src_root: Path, output_epub: Path) -> None:
    output_epub.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in src_root.rglob('*') if p.is_file()]
    files_sorted = sorted(files, key=lambda p: (p.name != 'mimetype', str(p)))
    with zipfile.ZipFile(output_epub, 'w') as zf:
        for p in files_sorted:
            arcname = str(p.relative_to(src_root)).replace('\\', '/')
            if p.name == 'mimetype':
                zf.write(p, arcname=arcname, compress_type=zipfile.ZIP_STORED)
            else:
                zf.write(p, arcname=arcname, compress_type=zipfile.ZIP_DEFLATED)

def translate_epub(input_epub: Path, output_epub: Path, target_lang: str, source_lang: Optional[str], model: str, api_key: Optional[str], context_file: Optional[Path]) -> None:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. pip install openai")
    context_text = context_file.read_text(encoding='utf-8') if context_file and context_file.exists() else None
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    if context_file and context_file.exists():
        logging.info('Context file loaded (%d chars): %s', len(context_text or ''), context_file)
        logging.info('Model: %s | Target: %s | Source: %s', model, target_lang, source_lang or 'auto')
    with tempfile.TemporaryDirectory() as td:
        root = safe_extract_epub(input_epub, Path(td))
        html_files = [p for p in root.rglob('*') if p.suffix.lower() in HTML_EXTS]
        for html_path in html_files:
            logging.info('Translating file: %s', html_path)
            translate_html_file(html_path, client, model, source_lang, target_lang, context_text)
        opf = find_opf(root)
        if opf:
            update_opf_language(opf, target_lang)
        repackage_epub(root, output_epub)

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Translate EPUB while preserving layout using OpenAI API.')
    p.add_argument('input_epub', type=Path)
    p.add_argument('output_epub', type=Path)
    p.add_argument('--target-lang', required=True)
    p.add_argument('--source-lang', default=None)
    p.add_argument('--context-file', type=Path, default=None, help='Optional .txt with glossary/context notes for translation')
    p.add_argument('--model', default='gpt-4o')
    p.add_argument('--openai-api-key', default=os.getenv('OPENAI_API_KEY'))
    p.add_argument('--verbose', action='store_true', help='Enable info-level logging')
    return p.parse_args(argv)

def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='[%(levelname)s] %(message)s'
    )
    logging.info('Starting translation: %s → %s', args.input_epub, args.output_epub)
    try:
        translate_epub(
            input_epub=args.input_epub,
            output_epub=args.output_epub,
            target_lang=args.target_lang,
            source_lang=args.source_lang,
            model=args.model,
            api_key=args.openai_api_key,
            context_file=args.context_file,
        )
        print(f"✅ Wrote translated EPUB to {args.output_epub}")
        return 0
    except Exception as e:
        print(f"❌ Fehler: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())

