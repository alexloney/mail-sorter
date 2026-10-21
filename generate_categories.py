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

from google.auth.exceptions import RefreshError
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
# Every email goes to the small, fast model first. If it comes back uncertain (see
# below), the email is re-classified by the larger model and that answer is used.
FAST_MODEL = "llama3.1:8b"
# MODEL = "odytrice/qwen3.8:4090-27b"
MODEL = "qwen-coder-64k:latest"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 300       # seconds before a single model call is abandoned
NUM_CTX = 8192             # plenty for ~3,000 chars of body + the system prompt
NUM_PREDICT = 1024         # thinking is disabled, so the JSON answer is all that's generated

# The fast model's answer is "uncertain" (and escalated to MODEL) when its confidence
# is in this set, when it picks one of these categories, or when its output is invalid.
ESCALATE_CONFIDENCE = {"medium", "low"}
ESCALATE_CATEGORIES = {"Needs Review"}
ESCALATE_IN_BODY_DOCUMENTS = True   # double-check "the email body is the document" answers

# The label to process, by its display name exactly as it appears in Gmail, e.g.
# "INBOX", "Promotions and Offers", or "Parent/Child" for a nested label.
# (Matching ignores upper/lower case. A raw ID such as "Label_88" also still works.)
LABEL_TO_PROCESS = "Inbox"

PAGE_SIZE = 500            # messages fetched per Gmail list request (max 500)
BATCH_SIZE = 20            # classified messages to collect before writing labels (1-1000)
MAX_BODY_CHARS = 3000      # how much of each email body the model sees

LABEL_ROOT = "AI"          # all labels are created as AI/<name>
ADD_YEAR_LABEL = True      # also add AI/Year/<year> (from Gmail's received date)
SENTIMENTAL_LABEL = f"{LABEL_ROOT}/Sentimental"

# Good candidates for Paperless-ngx get one of these on top of their category label.
PAPERLESS_LABEL = f"{LABEL_ROOT}/Paperless"
PAPERLESS_ATTACHMENTS_LABEL = f"{PAPERLESS_LABEL}/Attachments"   # the document is attached
PAPERLESS_EMAIL_LABEL = f"{PAPERLESS_LABEL}/Email"               # the email body is the document
PAPERLESS_REVIEW_LABEL = f"{PAPERLESS_LABEL}/Review"             # model said archive, but unsure
PAPERLESS_REVIEW_CONFIDENCE = {"low"}   # paperless confidence that goes to Review instead
MIN_IMAGE_ATTACHMENT_BYTES = 50_000     # smaller images are treated as logos/signatures
MAX_ATTACHMENTS_LISTED = 10             # file names shown to the model per email

# When True, classified messages are moved out of LABEL_TO_PROCESS (that label is
# removed). When False, every message only gets its AI/ labels and stays where it is.
MOVE_OUT_OF_SOURCE = True

# Messages matching any of these are labeled but left where they are for you to review
# (the label being processed is not removed from them).
KEEP_IN_INBOX_CATEGORIES = {"Direct Correspondence", "Needs Review"}
KEEP_SENTIMENTAL_IN_INBOX = True
KEEP_LOW_CONFIDENCE_IN_INBOX = True

# When False, already-labeled messages are skipped. When True, they are classified
# again and their old AI/<Category>, Sentimental and Paperless labels are replaced
# (AI/Year labels are kept). Useful for re-running to find Paperless candidates.
REPROCESS_MAIL = False

# When skipping, this query keeps most labeled messages out of the listing entirely;
# the label check inside process_message() is the safety net.
LIST_QUERY = None if REPROCESS_MAIL else f"-label:{LABEL_ROOT}"

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
        "and notices about those accounts. Not store receipts, and not bills or "
        "scheduled payments to service providers such as insurance premiums or toll accounts."
    ),
    "Health and Medical": (
        "Doctor and dental appointments, insurance claims, medical records, "
        "prescriptions, and patient portal messages. Not pet health."
    ),
    "Home and Admin": (
        "HOA communications, contractor quotes, tax documents, utility service "
        "changes or outages, local municipal correspondence, leases, and personal "
        "appointments. Not utility, trash, or other service bills."
    ),

    # --- Topic categories (beat Newsletters; see priority rules for Purchases/Promotions) ---
    "Pet Care": (
        "Vet records, pet health, and rescue adoption paperwork for "
        "Mister Buscits, Mira, and Kittles. Not orders, autoship, or shipping for pet supplies."
    ),
    "Travel and Itineraries": (
        "Vacation details, cruise excursions, campground reservations, "
        "flight tickets, hotel bookings, and rental cars."
    ),
    "Hobbies and Classes": (
        "Schedules and details for classes, studios, and clubs you belong to: dance studio "
        "schedules, your own performance showcase details, convention registrations, and "
        "hobby group or club mail. Not advertising from theaters, ballet companies, "
        "entertainment brands, or venues."
    ),

    # --- Generic automated mail ---
    "Purchases and Billing": (
        "Receipts, orders, autoship orders, delivery and shipping notifications (including "
        "FedEx, UPS, and USPS), meal kit credits, return confirmations, invoices, "
        "subscription charges, and bills or scheduled payments for services such as "
        "insurance premiums, toll accounts, utilities, and trash pickup."
    ),
    "Promotions and Offers": (
        "Marketing sales, discount codes, coupon alerts, retailer loyalty program "
        "updates, post-purchase survey or product review requests, and advertising for "
        "shows, performances, seasons, parks, and entertainment brands."
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
4. Decide what KIND of email it is before considering its topic:
   - Bills, scheduled or automatic payments, orders, autoship, receipts, and shipping or
     delivery notices go to Purchases and Billing, even when the topic is pets, insurance,
     home, utilities, or tolls. Exceptions: travel bookings stay in Travel and Itineraries,
     and bank or credit card statements stay in Finance and Banking.
   - Marketing and advertising go to Promotions and Offers, even when the topic matches a
     hobby (e.g. a ballet company or theme park promoting shows, tickets, or deals).
5. Otherwise, topic categories (Pet Care, Travel and Itineraries, Hobbies and Classes,
   Health and Medical, School and Work, Finance and Banking, Home and Admin)
   beat generic ones (Purchases and Billing, Promotions and Offers, Newsletters and Updates).
6. Use Junk and Dead Services for spam, chain mail, defunct services, and the
   throwaway account mail described in rule 2.
7. If still unclear, use Needs Review."""

PAPERLESS_RULES = """\
Separately from the category, decide whether the email belongs in Paperless-ngx, the owner's
document archive. It belongs there only if the email IS a document or carries one that may be
useful to find in the future:
- Financial: statements, tax forms (W-2, 1099, 1098), loan and mortgage paperwork, account
  opening and closure confirmations.
- Medical: visit summaries, lab results, EOBs (explanations of benefits), vaccination records.
- Insurance: policies, declarations pages, ID cards, claims.
- Home and vehicle: leases, closing documents, HOA documents, contractor invoices for work done,
  registrations, titles, repair records.
- Pets: vet records, vaccination certificates, adoption and microchip paperwork.
- Work, school and legal: offer letters, pay stubs, transcripts, signed contracts, official letters.
- Purchases: receipts for big purchases or anything under warranty.
Shipping and delivery notices, small orders, promotions, newsletters, and one-time codes do
not belong there.
Then say WHERE the document is, as paperless_document:
- 'attached': a document file appears on the 'Attachments' line. If that line says 'none',
  never answer 'attached'.
- 'in_body': the email text itself contains the document's actual details, e.g. a full
  receipt with items and prices, policy numbers and coverage, or the text of an official letter.
- 'only_via_link': the email only announces that a document is ready, available, or can be
  viewed or downloaded by clicking a link or logging in. Example: 'Your new statement is ready
  to be downloaded from your account' with no attachment is 'only_via_link', even though the
  statement itself would be worth keeping. Mentioning a document is not the same as containing it.
- 'none': no document worth keeping.
The 'Attachments' line lists the real attached files (inline images, logos and calendar invites
are already removed); use the file names as evidence."""

_DESCRIPTIONS_TEXT = "\n".join(f"- '{k}': {v}" for k, v in CATEGORY_DESCRIPTIONS.items())

SYSTEM_PROMPT = (
    "You are an exact and meticulous email categorizer. "
    "Classify the provided email into EXACTLY one of these categories, based on their definitions:\n"
    f"{_DESCRIPTIONS_TEXT}\n\n"
    "RULES:\n"
    "- Do not invent, suggest, or output any category name that is not strictly in the list above.\n"
    "- If you cannot determine a category, use 'Needs Review'.\n"
    "- 'Signals' are hints gathered from the email's headers and the owner's mailbox, not rules. "
    "Older labels are folders the owner created in the past; treat them as clues about the topic.\n\n"
    "PRIORITY RULES:\n"
    f"{PRIORITY_RULES}\n\n"
    "PAPERLESS RULES:\n"
    f"{PAPERLESS_RULES}\n\n"
    "OUTPUT FIELDS:\n"
    "- reason: one short sentence explaining the choice.\n"
    "- category: the category name.\n"
    "- sentimental: true only for personal or nostalgic email involving real people in the owner's "
    "life (friends, family, partners), or documents they would regret losing. Always false for automated mail.\n"
    "- confidence: high, medium, or low.\n"
    "- paperless_document: attached, in_body, only_via_link, or none (see PAPERLESS RULES).\n"
    "- paperless_confidence: high, medium, or low, for the paperless_document answer."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "category": {"type": "string", "enum": list(CATEGORY_DESCRIPTIONS)},
        "sentimental": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "paperless_document": {
            "type": "string", "enum": ["attached", "in_body", "only_via_link", "none"],
        },
        "paperless_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["reason", "category", "sentimental", "confidence",
                 "paperless_document", "paperless_confidence"],
}

# paperless_document answers that earn an AI/Paperless/... label.
ARCHIVE_DOCUMENTS = {"attached", "in_body"}

# Attachments with these types are calendar invites, not documents.
CALENDAR_TYPES = {"text/calendar", "application/ics"}

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
            try:
                creds.refresh(Request())
            except RefreshError as e:
                # Refresh token expired or revoked (e.g. OAuth app in "Testing"
                # mode expires them after 7 days) -- sign in again.
                print(f"Stored token could not be refreshed ({e}); re-authorizing...")
                creds = None
        else:
            creds = None
        if creds is None:
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
        self.source_id = None    # ID of the label being processed
        response = service.users().labels().list(userId="me").execute(num_retries=GMAIL_RETRIES)
        for label in response.get("labels", []):
            self.by_name[label["name"]] = label["id"]
            if label.get("type") == "user" and not self._is_ours(label["name"]):
                self.legacy_names[label["id"]] = label["name"]

    @staticmethod
    def _is_ours(name):
        return name == LABEL_ROOT or name.startswith(LABEL_ROOT + "/")

    def resolve(self, name):
        """Turn a label display name (or a raw ID) into a Gmail label ID."""
        if name in self.by_name:
            return self.by_name[name]
        lowered = {n.lower(): i for n, i in self.by_name.items()}
        if name.lower() in lowered:
            return lowered[name.lower()]
        if name in self.by_name.values():      # a raw ID like "Label_88" still works
            return name
        available = sorted(n for n in self.by_name if not self._is_ours(n))
        raise ValueError(
            f"No Gmail label named {name!r}. Names are matched exactly (ignoring case), "
            f"and nested labels use 'Parent/Child'.\nAvailable labels:\n  " + "\n  ".join(available)
        )

    def set_source(self, name):
        self.source_id = self.resolve(name)
        self.legacy_names.pop(self.source_id, None)   # not a useful hint for the model
        return self.source_id

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
        return any(
            self.by_name.get(f"{LABEL_ROOT}/{category}") in message_label_ids
            for category in CATEGORY_DESCRIPTIONS
        )

    def replaceable_ids(self):
        """IDs of the labels a reprocessing run replaces (everything we add except AI/Year)."""
        names = [f"{LABEL_ROOT}/{category}" for category in CATEGORY_DESCRIPTIONS]
        names += [SENTIMENTAL_LABEL, PAPERLESS_ATTACHMENTS_LABEL,
                  PAPERLESS_EMAIL_LABEL, PAPERLESS_REVIEW_LABEL]
        return {self.by_name[name] for name in names if name in self.by_name}

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


def list_attachments(parsed):
    """File names of real attachments, skipping inline images, small logos, and calendar invites."""
    found = []
    try:
        for part in parsed.walk():
            if part.is_multipart():
                continue
            content_type = part.get_content_type()
            filename = strip_invisible(part.get_filename() or "").strip()
            disposition = part.get_content_disposition()   # 'attachment', 'inline', or None
            if not filename and disposition != "attachment":
                continue                                    # body text, not a file
            if content_type in CALENDAR_TYPES or filename.lower().endswith((".ics", ".vcs")):
                continue
            if content_type.startswith("image/"):
                size = len(part.get_payload(decode=True) or b"")
                embedded = disposition != "attachment" and part["Content-ID"]
                if embedded or size < MIN_IMAGE_ATTACHMENT_BYTES:
                    continue
            found.append(filename or f"(unnamed {content_type})")
    except Exception:
        pass
    return found


def attachments_text(attachments):
    if not attachments:
        return "none"
    text = ", ".join(short(name, 80) for name in attachments[:MAX_ATTACHMENTS_LISTED])
    extra = len(attachments) - MAX_ATTACHMENTS_LISTED
    return text + (f" (+{extra} more)" if extra > 0 else "")


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


def classify_with(client, model, user_prompt):
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format=RESPONSE_SCHEMA,
        think=False,
        options={"temperature": 0.0, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
    )
    data = json.loads((response.message.content or "").strip())
    if data.get("category") not in CATEGORY_DESCRIPTIONS:
        raise ValueError(f"{model} returned an unknown category: {data.get('category')!r}")
    return data


def uncertainty_reason(data):
    """Why the fast model's answer should be escalated, or None if it is good enough."""
    if data["category"] in ESCALATE_CATEGORIES:
        return f"chose {data['category']}"
    if data.get("confidence") in ESCALATE_CONFIDENCE:
        return f"{data['category']} ({data['confidence']})"
    document = data.get("paperless_document")
    # Small models are rarely sure that a promo is NOT a document, so only a "yes" counts here.
    if document in ARCHIVE_DOCUMENTS and data.get("paperless_confidence") in ESCALATE_CONFIDENCE:
        return f"Paperless candidate ({data['paperless_confidence']})"
    # Body-as-document answers are rare and easy to get wrong (e.g. "statement ready" notices).
    if ESCALATE_IN_BODY_DOCUMENTS and document == "in_body":
        return "email body is the document"
    return None


def classify(client, user_prompt):
    """Try FAST_MODEL first; fall back to MODEL if it is uncertain or its output is invalid."""
    try:
        data = classify_with(client, FAST_MODEL, user_prompt)
    except (ValueError, ollama.ResponseError) as e:  # JSONDecodeError is a ValueError
        print(f"  {FAST_MODEL} failed ({short(str(e), 100)}) - escalating to {MODEL}.")
    else:
        reason = uncertainty_reason(data)
        if reason is None:
            data["model"] = FAST_MODEL
            return data
        print(f"  {FAST_MODEL} unsure: {reason} - escalating to {MODEL}.")

    data = classify_with(client, MODEL, user_prompt)
    data["model"] = MODEL
    return data


def should_stay_in_inbox(data):
    return (
        data["category"] in KEEP_IN_INBOX_CATEGORIES
        or (KEEP_SENTIMENTAL_IN_INBOX and data.get("sentimental", False))
        or (KEEP_LOW_CONFIDENCE_IN_INBOX and data.get("confidence") == "low")
    )


def paperless_label(data, attachments):
    """Which AI/Paperless/... label the message gets, or None."""
    document = data.get("paperless_document")
    if document not in ARCHIVE_DOCUMENTS:
        return None                                  # nothing here, or only behind a link
    if document == "attached" and not attachments:
        return PAPERLESS_REVIEW_LABEL                # model claims a file we couldn't find
    if data.get("paperless_confidence") in PAPERLESS_REVIEW_CONFIDENCE:
        return PAPERLESS_REVIEW_LABEL
    return PAPERLESS_ATTACHMENTS_LABEL if attachments else PAPERLESS_EMAIL_LABEL


def process_message(service, client, labels, msg_id):
    """Fetch and classify one message. Returns a pending-label dict, or None if skipped."""
    # One call: the raw format already includes labelIds and internalDate.
    msg = service.users().messages().get(userId="me", id=msg_id, format="raw").execute(
        num_retries=GMAIL_RETRIES
    )
    label_ids = set(msg.get("labelIds", []))
    if not REPROCESS_MAIL and labels.is_sorted(label_ids):
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
    attachments = list_attachments(parsed)
    attachment_line = attachments_text(attachments)

    print(f"  From:    {short(sender)}")
    print(f"  Subject: {short(subject)}")
    print(f"  Signals: {short(signals, 160)}")
    if attachments:
        print(f"  Attach:  {short(attachment_line, 160)}")

    user_prompt = (
        f"From: {sender}\n"
        f"Date: {date_text}\n"
        f"Subject: {subject}\n"
        f"Signals: {signals}\n"
        f"Attachments: {attachment_line}\n"
        f"Content:\n{body}\n"
    )
    data = classify(client, user_prompt)

    keep = not MOVE_OUT_OF_SOURCE or should_stay_in_inbox(data)
    sentimental = bool(data.get("sentimental", False))
    paperless = paperless_label(data, attachments)
    print(f"  Result:  {data['category']} ({data.get('confidence', '?')}, {data['model']})"
          f"{' + Sentimental' if sentimental else ''}"
          f"{f' + {paperless}' if paperless else ''} - {short(data.get('reason', ''), 150)}")
    if keep and MOVE_OUT_OF_SOURCE:
        print("           Staying where it is for your review.")
    print()

    return {
        "id": msg_id,
        "category": data["category"],
        "sentimental": sentimental,
        "paperless": paperless,
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
        groups[(item["category"], item["sentimental"], item["paperless"], item["keep"])].append(item)

    applied = []
    for (category, sentimental, paperless, keep), items in groups.items():
        ids = [item["id"] for item in items]
        add_names = [f"{LABEL_ROOT}/{category}"] + ([SENTIMENTAL_LABEL] if sentimental else [])
        if paperless:
            add_names.append(paperless)
        try:
            add_ids = [labels.get_or_create(name) for name in add_names]
            remove_ids = set() if keep else {labels.source_id}
            if REPROCESS_MAIL:
                remove_ids |= labels.replaceable_ids()      # drop the previous run's labels
            remove_ids -= set(add_ids)                      # Gmail rejects add+remove of one label
            failed = modify_messages(service, ids, add_ids, sorted(remove_ids))
        except HttpError as error:
            failed = set(ids)
            for msg_id in ids:
                log_failure(msg_id, f"could not create label: HTTP {error.resp.status}")

        done = [item for item in items if item["id"] not in failed]
        applied.extend(done)
        stats["labeled"] += len(done)
        stats["failed"] += len(failed)
        if done:
            note = " (left in place)" if keep and MOVE_OUT_OF_SOURCE else ""
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
    try:
        source_id = labels.set_source(LABEL_TO_PROCESS)
    except ValueError as error:
        print(error)
        return
    print(f"Processing label: {LABEL_TO_PROCESS} (id: {source_id})")
    if REPROCESS_MAIL:
        print("Reprocessing: already-labeled messages will be classified again.")
    print()

    labels.get_or_create(LABEL_ROOT)                # parent label so AI/... nests in the sidebar
    labels.get_or_create(PAPERLESS_LABEL)
    if ADD_YEAR_LABEL:
        labels.get_or_create(f"{LABEL_ROOT}/Year")

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
                labelIds=[source_id],
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
                    item = process_message(service, client, labels, ref["id"])
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

            # Removing the source label while paging may make Gmail's cursor skip a few
            # messages. Anything missed is still unlabeled, so the next run picks it up.
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
            print(f"Failures are logged in {FAILED_LOG}; they stay put and retry next run.")


if __name__ == "__main__":
    main()