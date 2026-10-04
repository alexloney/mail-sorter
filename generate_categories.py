"""
Gmail inbox categorizer (v2).

Walks the inbox page by page, asks a local Ollama model to categorize each email,
and applies nested AI/<Category> labels in batches. Messages that already carry an
AI/<Category> label are skipped, so the script is safe to stop and re-run.
"""
import base64
import email
import json
import os.path
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from email import policy

import ollama
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# gmail.modify already covers label operations; gmail.labels is kept only so your
# existing token.json stays valid.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
]

# ----------------------------- Settings ---------------------------------------
# MODEL = "odytrice/qwen3.8:4090-27b"
MODEL = "qwen-coder-64k:latest"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 300       # seconds before a single model call is abandoned
NUM_CTX = 8192             # plenty for ~3,000 chars of body + the system prompt
NUM_PREDICT = 10000        # thinking model requires a larger output context window

LABEL_TO_PROCESS = "Label_88" # "INBOX"
PAGE_SIZE = 500            # messages fetched per Gmail list request (max 500)
BATCH_SIZE = 20            # classified messages to collect before writing labels (1-1000)
MAX_BODY_CHARS = 3000      # how much of each email body the model sees

LABEL_ROOT = "AI"          # all labels are created as AI/<name>
ADD_YEAR_LABEL = True      # also add AI/Year/<year> (from Gmail's received date)
SENTIMENTAL_LABEL = f"{LABEL_ROOT}/Sentimental"

# Messages matching any of these are labeled but left in the inbox for you to review.
KEEP_IN_INBOX_CATEGORIES = {"Direct Correspondence", "Needs Review"}
KEEP_SENTIMENTAL_IN_INBOX = True
KEEP_LOW_CONFIDENCE_IN_INBOX = True

# Already-labeled messages are skipped. This query keeps most of them out of the
# listing entirely; the label check inside process_message() is the safety net.
LIST_QUERY = f"-label:{LABEL_ROOT}"

GMAIL_RETRIES = 5          # built-in exponential backoff for 429/5xx errors
MAX_CONSECUTIVE_FAILURES = 5   # stop if this many messages in a row fail
FAILED_LOG = "failed_messages.log"
# ------------------------------------------------------------------------------

CATEGORY_DESCRIPTIONS = {
    # --- Human and high-value (checked first) ---
    "Direct Correspondence": (
        "Emails written by a real person directly to you, including friends, family, "
        "and Aledyn. Personal conversations, not automated mail or mass forwards."
    ),
    "Account and Security": (
        "Lasting account records: welcome emails and account-created confirmations "
        "(proof an account exists), password-changed and recovery-info-changed notices, "
        "new-device and suspicious login alerts, and account closure confirmations. "
        "Not one-time codes or expiring links."
    ),
    "School and Work": (
        "Classes, teachers, assignments, school announcements, old jobs, "
        "job applications, resumes, and coworker or employer correspondence."
    ),
    "Finance and Banking": (
        "Bank and credit card statements, loans, investments, retirement accounts, "
        "and payment account notices. Not store receipts."
    ),
    "Health and Medical": (
        "Doctor and dental appointments, insurance claims, medical records, "
        "prescriptions, and patient portal messages. Not pet health."
    ),
    "Home and Admin": (
        "HOA communications, contractor quotes, tax documents, utilities, "
        "local municipal correspondence, leases, and personal appointments."
    ),

    # --- Topic categories (beat generic ones like Purchases/Newsletters) ---
    "Pet Care": (
        "Vet records, rescue adoption paperwork, and supplies for "
        "Mister Buscits, Mira, and Kittles."
    ),
    "Travel and Itineraries": (
        "Vacation details, cruise excursions, campground reservations, "
        "flight tickets, hotel bookings, and rental cars."
    ),
    "Hobbies and Classes": (
        "Dance studio schedules, performance showcase details, "
        "convention registrations, and hobby group or club mail."
    ),

    # --- Generic automated mail ---
    "Purchases and Billing": (
        "Receipts, delivery tracking, meal kit credits, return confirmations, "
        "invoices, and subscription charges."
    ),
    "Promotions and Offers": (
        "Marketing sales, discount codes, coupon alerts, retailer loyalty program "
        "updates, and post-purchase survey or product review requests."
    ),
    "Newsletters and Updates": (
        "Mailing lists, product feature updates, and corporate news blasts "
        "that you subscribed to or that contain real content."
    ),
    "Social Notifications": (
        "Automated social media pings, forum reply notifications, "
        "and generic event invitations from platforms."
    ),
    "System and Dev Alerts": (
        "Automated home server alerts, Docker/reverse proxy notifications, "
        "and GitHub pull request updates."
    ),

    # --- Bulk cleanup and fallback ---
    "Junk and Dead Services": (
        "Spam, chain emails, mass forwards, and mail from defunct or long-abandoned "
        "early-internet sites, forums, and services. Also throwaway account mail that "
        "is useless once used or expired: 2FA and one-time codes, password reset links, "
        "email verification and 'confirm your email' links, and magic sign-in links. "
        "Safe to bulk archive."
    ),
    "Needs Review": (
        "Unusual, ambiguous, or highly complex emails that do not clearly fit any "
        "category and require human sorting."
    ),
}

PRIORITY_RULES = """\
When an email fits more than one category, choose using this order:
1. Direct Correspondence beats everything. If a human wrote it to you personally, use it.
2. One-time codes and expiring links (2FA codes, password reset links, email verification
   links, magic sign-in links) go to Junk and Dead Services, even though they are account-related.
3. Account and Security beats all other automated categories.
4. Topic categories (Pet Care, Travel and Itineraries, Hobbies and Classes,
   Health and Medical, School and Work, Finance and Banking, Home and Admin)
   beat generic ones (Purchases and Billing, Promotions and Offers, Newsletters and Updates).
5. Use Junk and Dead Services for spam, chain mail, defunct services, and the
   throwaway account mail described in rule 2.
6. If still unclear, use Needs Review."""

# Gmail's own tab labels, translated for the prompt.
GMAIL_TABS = {
    "CATEGORY_PROMOTIONS": "Promotions",
    "CATEGORY_SOCIAL": "Social",
    "CATEGORY_UPDATES": "Updates",
    "CATEGORY_FORUMS": "Forums",
    "CATEGORY_PERSONAL": "Primary",
}

LABEL_STYLE = {"textColor": "#ffffff", "backgroundColor": "#d93025"}


class AbortRun(Exception):
    """Raised to stop the whole run (e.g. Ollama is down)."""


# ----------------------------- Auth -------------------------------------------
def get_creds():
    creds = None
    # token.json stores the access/refresh tokens and is created automatically the
    # first time the authorization flow completes.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds


# ----------------------------- Labels -----------------------------------------
class Labels:
    """Tracks Gmail label names/IDs and creates the nested AI/ labels on demand."""

    def __init__(self, service):
        self.service = service
        self.by_name = {}        # every label name -> id
        self.legacy_names = {}   # id -> name, for labels you made yourself (prompt hints)
        response = service.users().labels().list(userId="me").execute(num_retries=GMAIL_RETRIES)
        for label in response.get("labels", []):
            self.by_name[label["name"]] = label["id"]
            if label.get("type") == "user" and not self._is_ours(label["name"]):
                self.legacy_names[label["id"]] = label["name"]

    @staticmethod
    def _is_ours(name):
        return name == LABEL_ROOT or name.startswith(LABEL_ROOT + "/")

    def get_or_create(self, name):
        if name in self.by_name:
            return self.by_name[name]
        print(f"  Creating new label: {name}")
        body = {
            "name": name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show",
            "color": LABEL_STYLE,
        }
        created = self.service.users().labels().create(userId="me", body=body).execute(
            num_retries=GMAIL_RETRIES
        )
        self.by_name[name] = created["id"]
        return created["id"]

    def is_sorted(self, message_label_ids):
        """True if the message already has one of our AI/<Category> labels."""
        for category in CATEGORY_DESCRIPTIONS:
            label_id = self.by_name.get(f"{LABEL_ROOT}/{category}")
            if label_id and label_id in message_label_ids:
                return True
        return False

    def legacy_names_for(self, message_label_ids):
        return sorted(self.legacy_names[i] for i in message_label_ids if i in self.legacy_names)


# ----------------------------- Text helpers -----------------------------------
def short(text, limit=120):
    return text if len(text) <= limit else text[: limit - 3] + "..."


def strip_invisible(text):
    """Replace control and zero-width characters, but keep real non-ASCII text."""
    return "".join(
        " " if (unicodedata.category(ch) in ("Cc", "Cf") and ch != "\n") else ch
        for ch in text
    )


def clean_body(text):
    text = strip_invisible(text)
    text = re.sub(r"(?m)^[ \t]*>.*$", "", text)       # quoted reply text
    text = re.sub(r"https?://\S+", "<LINK>", text)
    text = re.sub(r"[^\S\n]+", " ", text)             # collapse spaces/tabs/nbsp
    text = re.sub(r"(?m)^[|\-: ]+$", "", text)        # leftover markdown table/divider rows
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def header_value(parsed, name):
    """Decoded, single-line header text ('' if missing or malformed)."""
    try:
        value = parsed[name]
        if value is None:
            return ""
        return " ".join(strip_invisible(str(value)).split())
    except Exception:
        return ""


def extract_body_text(parsed):
    """Best-effort readable text for the email body; '' if nothing usable."""
    try:
        part = parsed.get_body(preferencelist=("html", "plain"))
        if part is None:
            return ""
        try:
            content = part.get_content()          # honors the part's declared charset
        except Exception:                          # unknown charset, bad encoding, etc.
            content = (part.get_payload(decode=True) or b"").decode("utf-8", errors="replace")
        if part.get_content_type() == "text/html":
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup(["style", "script", "head", "title"]):
                tag.decompose()
            content = md(str(soup), strip=["a", "img"])   # keep link text, drop links/images
        return clean_body(content[: MAX_BODY_CHARS * 10])
    except Exception:
        return ""


def received_date(msg):
    try:
        return datetime.fromtimestamp(int(msg["internalDate"]) / 1000, tz=timezone.utc)
    except (KeyError, ValueError, OSError, OverflowError):
        return None


def build_signals(parsed, label_ids, labels):
    """Free hints from headers and Gmail labels that help the model."""
    signals = [
        f"Gmail filed it under its {GMAIL_TABS[lid]} tab"
        for lid in sorted(label_ids) if lid in GMAIL_TABS
    ]
    if "STARRED" in label_ids:
        signals.append("you starred it")
    if header_value(parsed, "List-Unsubscribe") or header_value(parsed, "List-Id"):
        signals.append("sent through a mailing list (has unsubscribe/list headers)")
    precedence = header_value(parsed, "Precedence").lower()
    if precedence in ("bulk", "list", "junk"):
        signals.append(f"Precedence header is '{precedence}'")
    auto = header_value(parsed, "Auto-Submitted").lower()
    if auto and auto != "no":
        signals.append("auto-generated message")
    legacy = labels.legacy_names_for(label_ids)
    if legacy:
        signals.append("already carries your older labels: " + ", ".join(legacy))
    return "; ".join(signals) if signals else "none"


# ----------------------------- Per-message work -------------------------------
def log_failure(msg_id, reason):
    print(f"  [FAILED] {msg_id}: {short(reason, 200)}")
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}\t{msg_id}\t{reason}\n")


def classify(client, system_prompt, schema, user_prompt):
    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format=schema,
        options={"temperature": 0.0, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
    )
    data = json.loads((response.message.content or "").strip())
    if data.get("category") not in CATEGORY_DESCRIPTIONS:
        raise ValueError(f"model returned an unknown category: {data.get('category')!r}")
    return data


def should_stay_in_inbox(data):
    return (
        data["category"] in KEEP_IN_INBOX_CATEGORIES
        or (KEEP_SENTIMENTAL_IN_INBOX and data.get("sentimental", False))
        or (KEEP_LOW_CONFIDENCE_IN_INBOX and data.get("confidence") == "low")
    )


def process_message(service, client, labels, system_prompt, schema, msg_id):
    """Fetch and classify one message. Returns a pending-label dict, or None if skipped."""
    # One call: the raw format already includes labelIds and internalDate.
    msg = service.users().messages().get(userId="me", id=msg_id, format="raw").execute(
        num_retries=GMAIL_RETRIES
    )
    label_ids = set(msg.get("labelIds", []))
    if labels.is_sorted(label_ids):
        print("  Already sorted - skipping.\n")
        return None

    raw = msg["raw"]
    raw_bytes = base64.urlsafe_b64decode((raw + "=" * (-len(raw) % 4)).encode("ASCII"))
    parsed = email.message_from_bytes(raw_bytes, policy=policy.default)

    sender = short(header_value(parsed, "From") or "(unknown sender)", 200)
    subject = short(header_value(parsed, "Subject") or "(no subject)", 300)
    date = received_date(msg)
    date_text = date.strftime("%Y-%m-%d") if date else "unknown"
    signals = build_signals(parsed, label_ids, labels)
    body = extract_body_text(parsed)[:MAX_BODY_CHARS] or "(no readable text)"

    print(f"  From:    {short(sender)}")
    print(f"  Subject: {short(subject)}")
    print(f"  Signals: {short(signals, 160)}")

    user_prompt = (
        f"From: {sender}\n"
        f"Date: {date_text}\n"
        f"Subject: {subject}\n"
        f"Signals: {signals}\n"
        f"Content:\n{body}\n"
    )
    data = classify(client, system_prompt, schema, user_prompt)

    keep = should_stay_in_inbox(data)
    sentimental = bool(data.get("sentimental", False))
    print(f"  Result:  {data['category']} ({data.get('confidence', '?')})"
          f"{' + Sentimental' if sentimental else ''} - {short(data.get('reason', ''), 150)}")
    if keep:
        print("           Staying in the inbox for your review.")
    print()

    return {
        "id": msg_id,
        "category": data["category"],
        "sentimental": sentimental,
        "keep": keep,
        "year": date.year if date else None,
    }


# ----------------------------- Writing labels ---------------------------------
def modify_messages(service, ids, add_ids, remove_ids):
    """
    Apply label changes with a single batchModify call. If Gmail rejects the batch
    (e.g. one problem message), retry one at a time so it can't block the rest.
    Returns the set of IDs that could not be modified.
    """
    labels_body = {"addLabelIds": add_ids, "removeLabelIds": remove_ids}
    messages = service.users().messages()
    try:
        messages.batchModify(userId="me", body={"ids": ids, **labels_body}).execute(
            num_retries=GMAIL_RETRIES
        )
        return set()
    except HttpError as error:
        if len(ids) == 1 or error.resp.status != 400:
            for msg_id in ids:
                log_failure(msg_id, f"batchModify failed: HTTP {error.resp.status}")
            return set(ids)
        print(f"  Batch rejected (HTTP 400); retrying {len(ids)} messages one at a time...")

    failed = set()
    for msg_id in ids:
        try:
            messages.modify(userId="me", id=msg_id, body=labels_body).execute(
                num_retries=GMAIL_RETRIES
            )
        except HttpError as error:
            failed.add(msg_id)
            log_failure(msg_id, f"modify failed: HTTP {error.resp.status}")
    return failed


def flush_batch(service, labels, pending, stats):
    """Write labels for every pending message, grouped so each group is one API call."""
    if not pending:
        return
    print(f"--- Applying labels to {len(pending)} message(s) ---")

    groups = defaultdict(list)
    for item in pending:
        groups[(item["category"], item["sentimental"], item["keep"])].append(item)

    applied = []
    for (category, sentimental, keep), items in groups.items():
        ids = [item["id"] for item in items]
        add_names = [f"{LABEL_ROOT}/{category}"] + ([SENTIMENTAL_LABEL] if sentimental else [])
        try:
            add_ids = [labels.get_or_create(name) for name in add_names]
            failed = modify_messages(service, ids, add_ids, [] if keep else [LABEL_TO_PROCESS])
        except HttpError as error:
            failed = set(ids)
            for msg_id in ids:
                log_failure(msg_id, f"could not create label: HTTP {error.resp.status}")

        done = [item for item in items if item["id"] not in failed]
        applied.extend(done)
        stats["labeled"] += len(done)
        stats["failed"] += len(failed)
        if done:
            note = " (kept in inbox)" if keep else ""
            print(f"  {len(done)} x {', '.join(add_names)}{note}")

    # Year labels use a separate call per year so they don't fragment the groups above.
    if ADD_YEAR_LABEL:
        by_year = defaultdict(list)
        for item in applied:
            if item["year"]:
                by_year[item["year"]].append(item["id"])
        for year, ids in sorted(by_year.items()):
            try:
                year_id = labels.get_or_create(f"{LABEL_ROOT}/Year/{year}")
                modify_messages(service, ids, [year_id], [])
            except HttpError as error:
                for msg_id in ids:
                    log_failure(msg_id, f"year label {year} failed: HTTP {error.resp.status}")

    print()
    pending.clear()


# ----------------------------- Main -------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # odd subject characters can't crash printing

    batch_size = max(1, min(BATCH_SIZE, 1000))
    client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
    service = build("gmail", "v1", credentials=get_creds())

    labels = Labels(service)
    labels.get_or_create(LABEL_ROOT)                # parent label so AI/... nests in the sidebar
    if ADD_YEAR_LABEL:
        labels.get_or_create(f"{LABEL_ROOT}/Year")

    # The categories are static, so the prompt and schema are built once.
    descriptions_text = "\n".join(f"- '{k}': {v}" for k, v in CATEGORY_DESCRIPTIONS.items())
    system_prompt = (
        "You are an exact and meticulous email categorizer. "
        "Classify the provided email into EXACTLY one of these categories, based on their definitions:\n"
        f"{descriptions_text}\n\n"
        "RULES:\n"
        "- Do not invent, suggest, or output any category name that is not strictly in the list above.\n"
        "- If you cannot determine a category, use 'Needs Review'.\n"
        "- 'Signals' are hints gathered from the email's headers and the owner's mailbox, not rules. "
        "Older labels are folders the owner created in the past; treat them as clues about the topic.\n\n"
        "PRIORITY RULES:\n"
        f"{PRIORITY_RULES}\n\n"
        "OUTPUT FIELDS:\n"
        "- reason: one short sentence explaining the choice.\n"
        "- category: the category name.\n"
        "- sentimental: true only for personal or nostalgic email involving real people in the owner's "
        "life (friends, family, partners), or documents they would regret losing. Always false for automated mail.\n"
        "- confidence: high, medium, or low."
    )
    schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "category": {"type": "string", "enum": list(CATEGORY_DESCRIPTIONS)},
            "sentimental": {"type": "boolean"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["reason", "category", "sentimental", "confidence"],
    }

    stats = {"seen": 0, "skipped": 0, "labeled": 0, "failed": 0}
    pending = []
    consecutive_failures = 0
    page_token = None
    page = 0

    try:
        while True:
            page += 1
            response = service.users().messages().list(
                userId="me",
                labelIds=[LABEL_TO_PROCESS],
                q=LIST_QUERY,
                maxResults=PAGE_SIZE,
                pageToken=page_token,
            ).execute(num_retries=GMAIL_RETRIES)

            messages = response.get("messages", [])
            if not messages:
                print("No more messages found.")
                break
            print(f"=== Page {page}: {len(messages)} messages ===\n")

            for index, ref in enumerate(messages, start=1):
                stats["seen"] += 1
                print(f"[Page {page} - {index}/{len(messages)}] {ref['id']}")
                try:
                    item = process_message(service, client, labels, system_prompt, schema, ref["id"])
                except Exception as error:   # one bad message shouldn't stop the run
                    stats["failed"] += 1
                    consecutive_failures += 1
                    log_failure(ref["id"], f"{type(error).__name__}: {error}")
                    print()
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        raise AbortRun(
                            f"{consecutive_failures} messages failed in a row - "
                            "check that Ollama is running and your network is up."
                        )
                    continue

                consecutive_failures = 0
                if item is None:
                    stats["skipped"] += 1
                    continue
                pending.append(item)
                if len(pending) >= batch_size:
                    flush_batch(service, labels, pending, stats)

            flush_batch(service, labels, pending, stats)   # finish the page before moving on

            # Archiving while paging may make Gmail's cursor skip a few messages.
            # Anything missed is still unlabeled, so the next run picks it up.
            page_token = response.get("nextPageToken")
            if not page_token:
                print("Reached the last page.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted - applying labels for messages already classified...")
    except AbortRun as error:
        print(f"Stopping: {error}")
    except HttpError as error:
        print(f"Gmail error while listing messages: {error}")
    finally:
        flush_batch(service, labels, pending, stats)
        print(
            f"Done. Seen {stats['seen']}, labeled {stats['labeled']}, "
            f"skipped {stats['skipped']} (already sorted), failed {stats['failed']}."
        )
        if stats["failed"]:
            print(f"Failures are logged in {FAILED_LOG}; they stay in the inbox and retry next run.")


if __name__ == "__main__":
    main()