import os.path
import base64
from markdownify import markdownify as md
import ollama
import json
import email
import re
import html

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels"
]

CACHE_FILE = "sender_cache.json"

def load_sender_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_sender_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

def get_creds():
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())
  return creds

def get_or_create_label(service, label_name, label_cache):
    if label_name in label_cache:
        return label_cache[label_name]
    
    print(f"  Creating new label: {label_name}")
    label_object = {
        "name": label_name,
        "labelListVisibility": "labelShow",
        "messageListVisibility": "show",
        "color": {
            "textColor": "#ffffff",
            "backgroundColor": "#d93025" 
        }
    }
    created_label = service.users().labels().create(userId="me", body=label_object).execute()
    label_cache[label_name] = created_label["id"]
    
    return created_label["id"]

def main():

    # Establish connection to Ollama and gmail
    client = ollama.Client(host="http://localhost:11434")
    model = "qwen-coder-64k:latest"
    creds = get_creds()
    service = build("gmail", "v1", credentials=creds)

    # Obtain a list of labels that already exist in gmail
    results = service.users().labels().list(userId="me").execute()
    existing_labels = results.get("labels", [])

    # Filter labels we don't want to include in the organization
    remove_labels = ['CHAT', 'SENT', 'INBOX', 'IMPORTANT', 'TRASH', 'DRAFT', 'SPAM', 'CATEGORY_FORUMS', 'CATEGORY_UPDATES', 'CATEGORY_PERSONAL', 'CATEGORY_PROMOTIONS', 'CATEGORY_SOCIAL', 'GREEN_CIRCLE', 'RED_CIRCLE', 'BLUE_STAR', 'YELLOW_STAR', 'STARRED', 'UNREAD', 'AA-Everything Else', 'Beth Edler', 'College', 'CS Club', 'Dating', 'Del', 'Flickr', 'Accounts', 'Friends', 'GameDev', 'Inner Circle Mail', 'Jeffrey', 'Jered', 'Jordan', 'loney.alex@gmail.com', 'Margaret', 'Mathematics', 'Melita', 'Mike', 'Mom', 'Namph', 'Orders', 'Other Stuff', 'PayPal', 'Photoshop', 'Programming Competition', 'PSC', 'Rachelle', 'Rodney', 'School', 'SMS', 'adeil.darkspell@gmail.com', 'UCA', 'Unsorted Stuff', 'FetLife', 'Alex Toney', 'Junk - Unsubscribe', 'Unsubscribe', 'Backups and stuff']
    label_cache = {}
    for label in existing_labels:
       if label["name"] not in remove_labels:
          label_cache[label["name"]] = label["id"]

    sender_cache = load_sender_cache()
    try:
        page_token = None

        while True:

            # Fetch the current page of up to 500 messages
            results = service.users().messages().list(
                userId="me", 
                labelIds=["INBOX"],
                maxResults=500,
                pageToken=page_token
            ).execute()
            
            messages = results.get("messages", [])

            if not messages:
                print("No more messages found.")
                break

            print(f"Processing batch of {len(messages)} messages...")

            # Process each returned message
            for message in messages:
                print(f'Message ID: {message["id"]}')

                # Obtain the message body from gmail, base64 decode it, then use email.message_from_bytes to parse it.
                msg = (service.users().messages().get(userId="me", id=message["id"]).execute())
                msg_body = service.users().messages().get(userId="me", id=message["id"], format="raw").execute()
                raw_bytes = base64.urlsafe_b64decode(msg_body['raw'].encode('ASCII'))
                parsed_email = email.message_from_bytes(raw_bytes)

                sender = parsed_email.get('From')


                # Use markdown library to attempt to parse it. However, if that fails, fallback to plain text
                extracted_text = ""
                for part in parsed_email.walk():
                    # Skip attachments and multipart containers
                    if part.get_content_maintype() == 'multipart' or part.get('Content-Disposition') is not None:
                        continue
                        
                    content_type = part.get_content_type()
                    try:
                        part_data = part.get_payload(decode=True).decode('utf-8', errors='replace')
                        if content_type == 'text/html':
                            extracted_text = md(part_data)
                            break  # HTML found, we can stop looking
                        elif content_type == 'text/plain':
                            extracted_text = part_data
                    except Exception:
                        continue

                # Clear links from the input prompt, saves context space
                extracted_text = re.sub(r'!\[.*?\]\([^\)]+\)', '', extracted_text)
                extracted_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', extracted_text)
                extracted_text = re.sub(r'https?://\S+', '<LINK>', extracted_text)

                snippet = html.unescape(msg.get("snippet", "No content"))
                safe_snippet = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', snippet)
                safe_snippet = re.sub(r' +', ' ', safe_snippet).strip()

                cleaned_text = extracted_text.strip()
                safe_body = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', cleaned_text)
                safe_body = re.sub(r' +', ' ', safe_body).strip()
                safe_body = re.sub(r'^[\|\-\s]+$', '', safe_body, flags=re.MULTILINE)
                safe_body = re.sub(r'\n+', '\n', safe_body).strip()

                user_prompt = (
                    f"Subject: {safe_snippet}\n"
                    f"Content: {safe_body[:10000]}\n"
                )
                print(f'  Subject: {safe_snippet}')

                if sender in sender_cache:
                    label_name = sender_cache[sender]
                    print(f'  [CACHE HIT] Categorized as: {label_name}')

                    label_id = get_or_create_label(service, label_name, label_cache)
                    if label_id:
                        service.users().messages().modify(
                            userId="me",
                            id=message["id"],
                            body={
                                "addLabelIds": [label_id],
                                "removeLabelIds": ["INBOX"]
                            }
                        ).execute()
                        print(f"  Moved to {label_name}.")
                else:
                    system_prompt = (
                        "You are a meticulous email categorizer auditing a single isolated email to determine its appropriate category. "
                        "You have a predefined set of categories: " + ", ".join(map(str, label_cache)) + ". "
                        "Carefully analyze the email content and assign it to the most appropriate category. "
                        "If none fit, suggest a new category name. "
                        "CRITICAL RULES FOR NEW CATEGORIES:\n"
                        "- Must be 1 to 3 words maximum.\n"
                        "- Do NOT include explanations, hyphens, or phrases like '(Suggested New Category)'.\n"
                        "- Examples of valid categories: 'Retail Offers', 'Shipping', 'Newsletters'.\n"
                        "If you cannot determine a category, output 'Needs Review'."
                    )

                    schema = {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string"
                        }
                    },
                    "required": ["category"]
                    }

                    response = client.chat(model=model, messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ], format=schema,
                    options={
                        "temperature": 0.0, # controls the randomness of the model's output
                        "num_ctx": 8192, # context window size
                        "num_predict": 10000 # maximum number of tokens to predict
                    })

                    # Parse the response from the categorization model
                    raw = (response.message.content or "").strip()
                    data = json.loads(raw)

                    print(f'  Categorized as: {data["category"]}')
                    category_name = data["category"]

                    # Get or create the Gmail label corresponding to the category
                    label_id = get_or_create_label(service, category_name, label_cache)
                    if label_id:
                        service.users().messages().modify(
                            userId="me",
                            id=message["id"],
                            body={
                                "addLabelIds": [label_id],
                                "removeLabelIds": ["INBOX"]
                            }
                        ).execute()
                        print(f"  Moved to {category_name}.")
                        sender_cache[sender] = category_name
                        save_sender_cache(sender_cache)
                print('')

    except HttpError as error:
        print(f"An error occured: {error}")


if __name__ == "__main__":
  main()