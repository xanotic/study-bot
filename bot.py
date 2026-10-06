"""
Study-notes Telegram bot
------------------------
1) SLIDES -> NOTES. Send a lecture slide (photo, PDF or PowerPoint .pptx) with the
subject code as the caption (e.g. "ISP560"). The bot:

  1. Sends the slide to Claude (images -> vision; PDFs -> per-page text, with
     scanned pages rendered to images - see pdf_fallback.py; PowerPoint -> slide
     text, speaker notes and pictures - see pptx_reader.py)
  2. Gets back Cornell-style markdown notes as JSON
  3. Saves the notes as a .md file in the matching Obsidian subject folder
     (links to existing notes in that folder are added as [[wikilinks]])

2) CHAT. Send any normal text message and the bot answers as a personal wiki: it
searches and reads your whole vault (see vault_chat.py) and can help with other things.
Use /reset to clear the conversation.

Runs locally with long polling - no public server needed.
"""
import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, NetworkError, TimedOut
from telegram.ext import Application, ContextTypes, MessageHandler, CommandHandler, filters

from pdf_fallback import MAX_PAGES, pdf_to_content_blocks
from pptx_reader import MAX_SLIDES, pptx_to_content_blocks
from vault_chat import VaultIndex, run_chat, trim_history

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
VAULT_PATH = Path(os.environ["VAULT_PATH"])
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
# Optional safety: only answer this Telegram user id (message @userinfobot to find yours)
ALLOWED_USER_ID = int(os.environ["ALLOWED_USER_ID"]) if os.getenv("ALLOWED_USER_ID") else None

MAX_IMAGE_BYTES = 4_500_000  # Claude's limit is 5 MB per image
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx INFO logs would print the bot token in URLs
log = logging.getLogger("study-bot")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
vault_index = VaultIndex(VAULT_PATH)
chat_histories: dict[int, list[dict]] = {}

PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

SYSTEM_PROMPT = """You are a study assistant for a university student. You turn lecture \
slides into clean Cornell-style notes.

Return ONLY a single JSON object (no prose, no code fences) with exactly these keys:
{
  "title": "short descriptive note title (no slashes or special characters)",
  "notes_markdown": "the Cornell notes as markdown (see format below)",
  "related": ["titles chosen ONLY from the provided existing_notes list that are genuinely related"]
}

notes_markdown format:
## Cues
- key questions / keywords, one per line, that the notes below answer
## Notes
- the main content: organised bullets, sub-bullets, definitions, formulas, small tables where useful.
  Preserve technical terms exactly as on the slide. Use LaTeX ($...$) for formulas.
## Summary
2-3 sentences capturing the main idea of the slide(s).

Rules:
- Do not invent facts that are not supported by the slide. If something is unreadable, say so in the notes.
- "related" must be [] if nothing fits. Never invent titles.
- Write in the same language as the slide."""


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def resolve_subject_folder(subject: str) -> Path:
    """Match the caption (e.g. 'ISP560') to a vault folder such as 'ISP560 - MACHINE LEARNING'.
    Falls back to creating a folder named after the code."""
    code = subject.strip().upper()
    VAULT_PATH.mkdir(parents=True, exist_ok=True)
    for child in sorted(VAULT_PATH.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            name = child.name.upper()
            if name == code or re.match(rf"^{re.escape(code)}(\b|\s|[-_:])", name):
                return child
    folder = VAULT_PATH / code
    folder.mkdir(exist_ok=True)
    return folder


def parse_subject(caption: str | None) -> str | None:
    """Pull a subject code like DSC551 out of the caption."""
    if not caption:
        return None
    m = re.search(r"\b([A-Za-z]{3}\s?\d{3})\b", caption)
    if m:
        return m.group(1).replace(" ", "").upper()
    first = caption.strip().split()[0] if caption.strip() else ""
    return first.upper() or None


def safe_filename(text: str) -> str:
    text = re.sub(r'[\\/:*?"<>|#^\[\]]', " ", text)
    text = re.sub(r"\s+", " ", text).strip().strip(".")
    return text[:80] or "Untitled"


def existing_note_titles(folder: Path, limit: int = 200) -> list[str]:
    notes = sorted(folder.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.stem for p in notes[:limit]]


def parse_json_reply(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def image_block(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large (over ~4.5 MB). Send a smaller photo or a PDF.")
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if media_type not in IMAGE_TYPES:
        media_type = "image/jpeg"
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": base64.standard_b64encode(data).decode()},
    }


def ask_claude(content: list[dict], subject: str, candidates: list[str]) -> dict:
    instruction = (
        f"Subject code: {subject}\n"
        f"existing_notes (for the 'related' field): {json.dumps(candidates, ensure_ascii=False)}\n"
        "Turn the lecture material above into notes. Return JSON only."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=6000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content + [{"type": "text", "text": instruction}]}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    data = parse_json_reply(raw)
    data.setdefault("title", f"{subject} slide")
    data.setdefault("notes_markdown", "")
    data.setdefault("related", [])
    return data


def save_note(folder: Path, subject: str, data: dict, source_name: str) -> Path:
    now = datetime.now()
    title = safe_filename(data["title"])
    path = folder / f"{title}.md"
    n = 2
    while path.exists():
        path = folder / f"{title} ({n}).md"
        n += 1

    known = {t.lower(): t for t in existing_note_titles(folder)}
    links = []
    for rel in data.get("related", []):
        real = known.get(str(rel).lower())
        if real and real != path.stem and f"[[{real}]]" not in links:
            links.append(f"[[{real}]]")

    body = [
        "---",
        f"subject: {subject}",
        f"date: {now:%Y-%m-%d}",
        f"source: {source_name}",
        "tags: [lecture, slides]",
        "---",
        f"# {data['title']}",
        "",
        data["notes_markdown"].strip(),
    ]
    if links:
        body += ["", "## Related", " · ".join(links)]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
# Telegram handlers
# ----------------------------------------------------------------------------
async def gate(update: Update) -> bool:
    """Only the owner may use the bot (it can read the whole vault and spends API credit)."""
    user = update.effective_user
    uid = user.id if user else None
    if ALLOWED_USER_ID is None:
        if update.message:
            await update.message.reply_text(
                "One-time setup: add this line to your .env file, then restart the bot:\n\n"
                f"ALLOWED_USER_ID={uid}\n\n"
                "This stops anyone else who finds the bot from reading your vault."
            )
        return False
    return uid == ALLOWED_USER_ID


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await gate(update):
        return
    await update.message.reply_text(
        "Notes: send a lecture slide (photo, PDF or PowerPoint .pptx) with the subject code as the caption, "
        "e.g. ISP560. I'll save Cornell notes into your vault.\n\n"
        "Chat: just type a message and I'll search your vault to answer - ask about a topic, get a quiz, "
        "a summary, a revision plan, anything.\n\n"
        "/reset clears our conversation."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await gate(update):
        return
    chat_histories.pop(update.message.chat_id, None)
    await update.message.reply_text("Conversation cleared.")


async def handle_slide(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not await gate(update):
        return

    subject = parse_subject(msg.caption)
    if not subject:
        await msg.reply_text("Please resend with the subject code as the caption (e.g. ISP560).")
        return

    # work out what we received
    if msg.photo:
        file_obj, kind, filename = msg.photo[-1], "image", f"slide_{msg.message_id}.jpg"
    elif msg.document:
        doc = msg.document
        mime = doc.mime_type or ""
        name_l = (doc.file_name or "").lower()
        if mime == "application/pdf" or name_l.endswith(".pdf"):
            kind = "pdf"
        elif mime == PPTX_MIME or name_l.endswith(".pptx"):
            kind = "pptx"
        elif name_l.endswith(".ppt"):
            await msg.reply_text("Old .ppt files aren't supported. Open it in PowerPoint, Save As .pptx, and resend.")
            return
        elif mime in IMAGE_TYPES:
            kind = "image"
        else:
            await msg.reply_text("I can read photos, PDFs and PowerPoint (.pptx) files.")
            return
        file_obj, filename = doc, doc.file_name or f"slide_{msg.message_id}"
    else:
        return

    try:
        tg_file = await file_obj.get_file()
    except BadRequest as e:
        if "too big" in str(e).lower():
            await msg.reply_text("Telegram only lets bots download files up to 20 MB. Compress it or split it and resend.")
            return
        raise

    label = {"image": "image", "pdf": "PDF", "pptx": "PowerPoint"}[kind]
    status = await msg.reply_text(f"Got it - reading your {label} for {subject}...")

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / safe_filename(filename)
        await tg_file.download_to_drive(custom_path=local)

        try:
            folder = resolve_subject_folder(subject)
            note = ""
            if kind == "image":
                content = [image_block(local)]
            elif kind == "pdf":
                content, stats = await asyncio.to_thread(pdf_to_content_blocks, str(local))
                note = f" ({stats['text_pages']} text pages, {stats['image_pages']} scanned pages"
                if stats["truncated"]:
                    note += f"; only the first {MAX_PAGES} of {stats['pages']} pages were read"
                note += ")"
                if not content:
                    await status.edit_text("That PDF has no pages I can read.")
                    return
            else:
                content, stats = await asyncio.to_thread(pptx_to_content_blocks, str(local))
                note = f" ({stats['slides']} slides, {stats['images']} pictures"
                if stats["truncated"]:
                    note += f"; only the first {MAX_SLIDES} slides were read"
                note += ")"
                if not content:
                    await status.edit_text("That PowerPoint has no slides I can read.")
                    return

            data = await asyncio.to_thread(ask_claude, content, subject, existing_note_titles(folder))
            path = save_note(folder, subject, data, filename)
        except ValueError as e:
            await status.edit_text(str(e))
            return
        except (json.JSONDecodeError, anthropic.APIError) as e:
            log.exception("Claude step failed")
            await status.edit_text(f"Claude step failed: {e}\nTry again in a moment.")
            return
        except Exception as e:  # noqa: BLE001
            log.exception("Unexpected failure")
            await status.edit_text(f"Something went wrong: {e}")
            return

    lines = [f"Saved: {folder.name}/{path.name}{note}"]
    if data["related"]:
        lines.append("Linked to: " + ", ".join(str(r) for r in data["related"][:5]))
    await status.edit_text("\n".join(lines))


def split_message(text: str, limit: int = 4000) -> list[str]:
    """Telegram caps messages at 4096 characters - split on paragraph/line boundaries."""
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


async def keep_typing(bot, chat_id: int, stop: asyncio.Event):
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:  # noqa: BLE001
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=4)
        except asyncio.TimeoutError:
            pass


async def handle_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not await gate(update):
        return
    text = (msg.text or "").strip()
    if not text:
        return

    chat_id = msg.chat_id
    history = chat_histories.get(chat_id, [])
    stop = asyncio.Event()
    typing = asyncio.create_task(keep_typing(context.bot, chat_id, stop))
    try:
        reply = await asyncio.to_thread(run_chat, client, MODEL, vault_index, history, text)
    except anthropic.APIError as e:
        log.exception("Claude chat failed")
        reply = f"Claude error: {e}\nTry again in a moment."
    except Exception as e:  # noqa: BLE001
        log.exception("Chat failed")
        reply = f"Something went wrong: {e}"
    else:
        chat_histories[chat_id] = trim_history(
            history + [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        )
    finally:
        stop.set()
        await typing

    for part in split_message(reply):
        await msg.reply_text(part)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Network drops are retried automatically by the library - just log one line."""
    if isinstance(context.error, (NetworkError, TimedOut)):
        log.warning("Network hiccup talking to Telegram (%s) - retrying automatically.", type(context.error).__name__)
    else:
        log.error("Unhandled error", exc_info=context.error)


def main():
    builder = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .get_updates_connect_timeout(30)
        .get_updates_read_timeout(30)
    )
    # Optional: route Telegram traffic through a proxy (e.g. http://127.0.0.1:7890 from a VPN app)
    proxy = os.getenv("TELEGRAM_PROXY")
    if proxy:
        builder = builder.proxy(proxy).get_updates_proxy(proxy)
        log.info("Using proxy for Telegram: %s", proxy)
    app = builder.build()
    app.add_error_handler(on_error)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, handle_slide))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_chat))
    log.info("Bot running (vault: %s, model: %s). Ctrl+C to stop.", VAULT_PATH, MODEL)
    # bootstrap_retries=-1: keep retrying the first connection instead of exiting on one network drop
    app.run_polling(bootstrap_retries=-1, drop_pending_updates=False)


if __name__ == "__main__":
    main()
