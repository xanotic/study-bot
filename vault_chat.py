"""
vault_chat.py - chat with your Obsidian vault like a personal wiki.

Claude gets tools to search, list and read your notes (BM25 keyword search over note
sections, so no extra services or embeddings needed), and - only when you ask - to
create a new note or append to an existing one. Nothing is ever overwritten or deleted,
and every path is locked inside the vault folder.
"""
import json
import logging
import math
import re
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

SKIP_DIRS = {".obsidian", ".trash", ".git"}
CHUNK_CHARS = 1500
MAX_READ_CHARS = 20_000
MAX_TOOL_ROUNDS = 8
HISTORY_MESSAGES = 20  # last N user/assistant messages remembered per chat


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]


def chunk_note(text: str) -> list[tuple[str, str]]:
    """Split a note into (heading, text) chunks of roughly CHUNK_CHARS."""
    sections: list[tuple[str, str]] = []
    heading, buf = "", []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if buf:
                sections.append((heading, "\n".join(buf)))
                buf = []
            heading = line.lstrip("#").strip()
        buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf)))

    chunks: list[tuple[str, str]] = []
    for h, body in sections:
        body = body.strip()
        while len(body) > CHUNK_CHARS * 1.5:
            cut = body.rfind("\n", 0, CHUNK_CHARS)
            if cut < CHUNK_CHARS // 2:
                cut = CHUNK_CHARS
            chunks.append((h, body[:cut].strip()))
            body = body[cut:].strip()
        if body:
            chunks.append((h, body))
    return chunks


class VaultIndex:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._lock = threading.Lock()
        self._files: dict[str, tuple[float, list[tuple[str, str]]]] = {}
        self._docs: list[dict] = []
        self._df: Counter = Counter()
        self._avgdl = 1.0

    # ---- indexing -----------------------------------------------------------
    def _iter_notes(self):
        for p in self.root.rglob("*.md"):
            rel = p.relative_to(self.root)
            if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
                continue
            yield p, rel.as_posix()

    def refresh(self) -> None:
        with self._lock:
            seen, changed = set(), not self._docs and not self._files
            for p, key in self._iter_notes():
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                seen.add(key)
                if key not in self._files or self._files[key][0] != mtime:
                    try:
                        text = p.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    self._files[key] = (mtime, chunk_note(text))
                    changed = True
            for key in list(self._files):
                if key not in seen:
                    del self._files[key]
                    changed = True
            if changed:
                self._rebuild()

    def _rebuild(self) -> None:
        docs, df = [], Counter()
        for path, (_, chunks) in self._files.items():
            title_tokens = tokenize(path.replace("/", " ").rsplit(".", 1)[0])
            for heading, text in chunks:
                toks = tokenize(text) + tokenize(heading) * 2 + title_tokens * 3
                tf = Counter(toks)
                docs.append({"path": path, "heading": heading, "text": text, "tf": tf, "len": len(toks)})
                df.update(tf.keys())
        self._docs, self._df = docs, df
        self._avgdl = (sum(d["len"] for d in docs) / len(docs)) if docs else 1.0

    # ---- tools --------------------------------------------------------------
    def search(self, query: str, folder: str | None = None, limit: int = 8) -> list[dict]:
        q = tokenize(query)
        if not q or not self._docs:
            return []
        n, k1, b = len(self._docs), 1.5, 0.75
        folder_l = (folder or "").lower().strip().replace("\\", "/")
        scored = []
        for d in self._docs:
            if folder_l and folder_l not in d["path"].lower():
                continue
            score = 0.0
            for term in set(q):
                f = d["tf"].get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * d["len"] / self._avgdl))
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)

        results, per_note = [], Counter()
        for score, d in scored:
            if per_note[d["path"]] >= 3:
                continue
            per_note[d["path"]] += 1
            results.append({
                "path": d["path"],
                "section": d["heading"],
                "score": round(score, 2),
                "excerpt": d["text"][:700],
            })
            if len(results) >= max(1, min(limit, 15)):
                break
        return results

    def _safe_path(self, rel: str) -> Path:
        rel = (rel or "").strip().replace("\\", "/").lstrip("/")
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError("That path is outside the vault.")
        parts = p.relative_to(self.root).parts
        if any(part.startswith(".") for part in parts):
            raise ValueError("Hidden folders are off-limits.")
        return p

    def _find_note(self, rel: str) -> Path:
        rel = rel.strip()
        candidates = [rel, rel + ".md"] if not rel.lower().endswith(".md") else [rel]
        for c in candidates:
            p = self._safe_path(c)
            if p.is_file():
                return p
        # fall back to a case-insensitive match on the note title (like an Obsidian [[link]])
        stem = Path(rel).stem.lower()
        matches = [p for p, key in self._iter_notes() if p.stem.lower() == stem]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError("Several notes have that title: " + ", ".join(m.relative_to(self.root).as_posix() for m in matches[:5]))
        raise ValueError(f"No note found at '{rel}'. Use list_notes or search_notes to find the right path.")

    def read_note(self, path: str) -> str:
        p = self._find_note(path)
        text = p.read_text(encoding="utf-8", errors="ignore")
        rel = p.relative_to(self.root).as_posix()
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + f"\n\n[... truncated, note is {len(text)} characters]"
        return f"# {rel}\n\n{text}"

    def list_notes(self, folder: str = "") -> str:
        base = self._safe_path(folder)
        if not base.is_dir():
            raise ValueError(f"'{folder}' is not a folder.")
        dirs, notes = [], []
        for child in sorted(base.iterdir(), key=lambda c: c.name.lower()):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                count = sum(1 for _ in child.rglob("*.md"))
                dirs.append(f"[folder] {child.relative_to(self.root).as_posix()}/  ({count} notes)")
            elif child.suffix.lower() == ".md":
                notes.append(f"{child.relative_to(self.root).as_posix()}")
        out = dirs + notes
        return "\n".join(out[:300]) if out else "(empty)"

    def create_note(self, path: str, content: str) -> str:
        rel = path.strip()
        if not rel.lower().endswith(".md"):
            rel += ".md"
        p = self._safe_path(rel)
        if p.exists():
            raise ValueError("A note with that path already exists. Use append_to_note, or pick another name.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content.strip() + "\n", encoding="utf-8")
        return f"Created {p.relative_to(self.root).as_posix()}"

    def append_to_note(self, path: str, content: str) -> str:
        p = self._find_note(path)
        with p.open("a", encoding="utf-8") as f:
            f.write("\n\n" + content.strip() + "\n")
        return f"Appended to {p.relative_to(self.root).as_posix()}"

    def top_folders(self) -> list[str]:
        return sorted(c.name for c in self.root.iterdir() if c.is_dir() and not c.name.startswith("."))


TOOLS = [
    {
        "name": "search_notes",
        "description": "Keyword search across all notes in the vault. Returns the best matching sections with a path "
                       "and excerpt. Use short, specific keywords (terms likely to appear in the notes), and try "
                       "several searches with different wording if the first finds little.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to search for"},
                "folder": {"type": "string", "description": "Optional: only search paths containing this text, e.g. a subject code like ISP560"},
                "limit": {"type": "integer", "description": "Max results (default 8, max 15)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_note",
        "description": "Read the full text of one note. Accepts a vault-relative path or just the note title.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "name": "list_notes",
        "description": "List subfolders and notes inside a vault folder (empty string = vault root).",
        "input_schema": {"type": "object", "properties": {"folder": {"type": "string"}}},
    },
    {
        "name": "create_note",
        "description": "Create a NEW markdown note. Only use when the user asks to save something. Fails if the note exists. "
                       "Path is vault-relative, e.g. 'ISP560 - MACHINE LEARNING/Exam revision.md'.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string", "description": "Markdown content"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "append_to_note",
        "description": "Append markdown to the end of an existing note. Only use when the user asks for it.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
]


def execute_tool(index: VaultIndex, name: str, args: dict) -> str:
    try:
        if name == "search_notes":
            res = index.search(args.get("query", ""), args.get("folder"), int(args.get("limit") or 8))
            return json.dumps(res, ensure_ascii=False) if res else "No matching notes. Try different keywords."
        if name == "read_note":
            return index.read_note(args["path"])
        if name == "list_notes":
            return index.list_notes(args.get("folder", ""))
        if name == "create_note":
            return index.create_note(args["path"], args["content"])
        if name == "append_to_note":
            return index.append_to_note(args["path"], args["content"])
        return f"Unknown tool: {name}"
    except (ValueError, KeyError, OSError) as e:
        return f"Error: {e}"


def build_system(index: VaultIndex) -> str:
    folders = ", ".join(index.top_folders()) or "(none yet)"
    return f"""You are the user's personal wiki assistant, chatting through Telegram. The user is a university student and \
keeps their study notes in an Obsidian vault. You can search, list and read every note in it with your tools.

Today is {datetime.now():%A %d %B %Y}.
Top-level vault folders (one per subject): {folders}

How to work:
- For anything about the user's courses, notes or past work, search the vault first, then read the most relevant notes in \
full when excerpts aren't enough. Try a couple of different keyword searches before concluding something isn't there.
- Ground answers in the notes and cite them as [[Note Title]]. If the notes don't cover something, say so plainly, \
then (if helpful) answer from general knowledge and label it clearly as not from their notes.
- You can also help with other things using the notes as material: explain concepts, quiz them, summarise a subject, \
build revision plans, compare topics, draft answers. Be accurate; don't invent content the notes don't contain.
- Only create or append to notes when the user asks you to save something. Never overwrite or delete. \
New notes go in the matching subject folder.
- Replies are shown as plain text in Telegram: keep them concise, use short paragraphs and simple dashes for lists, \
and avoid markdown tables, headings and heavy formatting."""


def run_chat(client, model: str, index: VaultIndex, history: list[dict], user_text: str) -> str:
    """One user turn. `history` is a list of plain {"role","content": str} messages."""
    index.refresh()
    system = build_system(index)
    messages = list(history) + [{"role": "user", "content": user_text}]

    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.messages.create(model=model, max_tokens=2500, system=system, tools=TOOLS, messages=messages)
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return text or "(I don't have an answer for that.)"

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                log.info("tool %s %s", block.name, json.dumps(block.input, ensure_ascii=False)[:200])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": execute_tool(index, block.name, block.input),
                })
        messages.append({"role": "user", "content": results})
        index.refresh()  # pick up notes the model just created/appended

    return "I looked through a lot of notes but couldn't wrap that up. Try a narrower question."


def trim_history(history: list[dict]) -> list[dict]:
    history = history[-HISTORY_MESSAGES:]
    while history and history[0]["role"] != "user":
        history = history[1:]
    return history
