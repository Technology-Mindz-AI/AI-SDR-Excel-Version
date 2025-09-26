import uuid
import sqlite3
import requests
import time
import threading
from typing import Optional
import re
from datetime import datetime
from config import settings
from requests.auth import HTTPBasicAuth
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Response, Header
from logger_config import logger
import pandas as pd
import io
import math
from pydantic import BaseModel
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Depends, HTTPException, Cookie, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from typing import Optional
from fastapi.responses import JSONResponse
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.query import Query
from appwrite.exception import AppwriteException
from helperfuncs import (
    CallRequest,
    QueueUpdateRequest,
    pop_next_call,
    update_call_details,
    pop_call_by_id,
    generate_initial_message,
    COUNTRY_CODE_MAP,
    init_db,
    DB_PATH
)
from notes_and_tasks import (
    summarize_conversation_transcript,
    update_customer_data_notes_and_tasks,
    export_customer_data_to_excel,
    send_meeting_invite
)

app = FastAPI(title="Call Queue",
              # docs_url=None,       # disable /docs
              # redoc_url=None,      # disable /redoc
              # openapi_url=None     # disable /openapi.json
              )


active_sessions = {}

# ==== Vapi.ai configuration ====
VAPI_BASE_URL = getattr(settings, "VAPI_BASE_URL",
                        "https://api.vapi.ai").rstrip("/")
VAPI_API_KEY = getattr(settings, "VAPI_API_KEY", None)
VAPI_ASSISTANT_ID = getattr(
    settings, "VAPI_ASSISTANT_ID", getattr(settings, "AGENT_ID", None))
VAPI_PHONE_NUMBER_ID = getattr(settings, "VAPI_PHONE_NUMBER_ID", getattr(
    settings, "AGENT_PHONE_NUMBER_ID", None))

account_sid = getattr(settings, "TWILIO_ACCOUNT_SID", None)

APPWRITE_ENDPOINT = getattr(settings, "APPWRITE_ENDPOINT", None)
APPWRITE_PROJECT_ID = getattr(settings, "APPWRITE_PROJECT_ID", None)
APPWRITE_API_KEY = getattr(settings, "APPWRITE_API_KEY", None)
APPWRITE_DATABASE_ID = getattr(settings, "APPWRITE_DATABASE_ID", None)
APPWRITE_COLLECTION_ID = getattr(settings, "APPWRITE_COLLECTION_ID", None)

# APPWRITE CLIENT INITIALIZATION
appwrite_client = Client()
appwrite_client.set_endpoint(APPWRITE_ENDPOINT)
appwrite_client.set_project(APPWRITE_PROJECT_ID)
appwrite_client.set_key(APPWRITE_API_KEY)

databases = Databases(appwrite_client)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- Vapi API interaction functions ---

def vapi_start_workflow_call(to_number: str, workflow_id: str, variables: dict) -> dict:
    """
    Start an outbound call using a WORKFLOW instead of assistant
    """
    if not VAPI_API_KEY:
        raise RuntimeError("VAPI_API_KEY missing in settings.")

    url = f"{VAPI_BASE_URL}/call/phone"
    payload = {
        "workflowId": workflow_id,  # Your workflow ID from Vapi dashboard
        "phoneNumberId": VAPI_PHONE_NUMBER_ID,
        "customer": {"number": to_number},
        "metadata": variables  # Same metadata you were using before!
    }

    headers = {"Authorization": f"Bearer {VAPI_API_KEY}"}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        raise RuntimeError(f"Vapi workflow call failed: {e}") from e


def vapi_start_outbound_call(
    to_number: str,
    metadata: dict,
    assistant_overrides: Optional[dict] = None
) -> dict:
    """
    Create an outbound call via Vapi API.
    Sends dynamic variables as metadata to make them available to the agent and webhook.
    """
    if not VAPI_API_KEY:
        raise RuntimeError("VAPI_API_KEY missing in settings.")
    if not VAPI_ASSISTANT_ID:
        raise RuntimeError(
            "VAPI_ASSISTANT_ID (or AGENT_ID) missing in settings.")
    if not VAPI_PHONE_NUMBER_ID:
        raise RuntimeError(
            "VAPI_PHONE_NUMBER_ID (or AGENT_PHONE_NUMBER_ID) missing in settings.")

    url = f"{VAPI_BASE_URL}/call/phone"

    # Get base URL for webhook from settings or construct from request base URL
    webhook_base_url = getattr(settings, "WEBHOOK_BASE_URL", None)
    if not webhook_base_url:
        logger.warning(
            "[vapi_start_outbound_call] WEBHOOK_BASE_URL not configured in settings")

    payload = {
        "assistantId": VAPI_ASSISTANT_ID,
        "phoneNumberId": VAPI_PHONE_NUMBER_ID,
        "customer": {"number": to_number},  # E.164 format expected
        "metadata": metadata or {}
    }

    if assistant_overrides:
        payload["assistantOverrides"] = assistant_overrides

    headers = {"Authorization": f"Bearer {VAPI_API_KEY}"}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        logger.error(
            f"[vapi_start_outbound_call] HTTP Error {resp.status_code}: {resp.text}")
        raise e
    except Exception as e:
        logger.error(f"[vapi_start_outbound_call] Unexpected error: {e}")
        raise e


def vapi_get_call_details(vapi_call_id: str, max_retries: int = 5, initial_delay: float = 5.0) -> Optional[dict]:
    """
    Fetches details of a Vapi call (summary, transcript, metadata) by ID with retries.
    Waits for call to be fully processed with transcript available.
    """
    if not VAPI_API_KEY:
        logger.error(
            "[vapi_get_call_details] Missing VAPI_API_KEY in settings.")
        return None

    url = f"{VAPI_BASE_URL}/call/{vapi_call_id}"
    headers = {"Authorization": f"Bearer {VAPI_API_KEY}"}

    logger.info(
        f"[vapi_get_call_details] Attempting to fetch call details from URL: {url}")

    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                f"[vapi_get_call_details] Attempt {attempt} of {max_retries} for call {vapi_call_id}")
            resp = requests.get(url, headers=headers, timeout=15)

            logger.info(
                f"[vapi_get_call_details] Response status code: {resp.status_code}")

            if resp.status_code == 200:
                data = resp.json()
                logger.info(
                    f"[vapi_get_call_details] Successfully retrieved data. Keys present: {list(data.keys())}")

                # Check if transcript is available
                transcript_available = False
                summary_available = False

                # Check various locations for transcript
                if data.get('transcript'):
                    transcript_available = True
                elif data.get('messages') and isinstance(data['messages'], list) and len(data['messages']) > 0:
                    transcript_available = True
                elif data.get('recording', {}).get('transcript'):
                    transcript_available = True
                elif data.get('analysis', {}).get('transcript'):
                    transcript_available = True

                # Check for summary
                if data.get('summary'):
                    summary_available = True
                elif data.get('analysis', {}).get('summary'):
                    summary_available = True
                elif data.get('analysis', {}).get('transcript_summary'):
                    summary_available = True

                logger.info(
                    f"[vapi_get_call_details] Transcript available: {transcript_available}, Summary available: {summary_available}")

                # If we have both transcript and summary, return the data
                if transcript_available and summary_available:
                    logger.info(
                        f"[vapi_get_call_details] Both transcript and summary available after {attempt} attempts")
                    return data
                else:
                    logger.info(
                        f"[vapi_get_call_details] Transcript/summary not yet available. Waiting {delay} seconds before next attempt")

            else:
                logger.error(
                    f"[vapi_get_call_details] Failed with status {resp.status_code}. Response: {resp.text}")

        except Exception as e:
            logger.error(
                f"[vapi_get_call_details] Attempt {attempt} failed for {vapi_call_id}: {e}")

        if attempt < max_retries:
            logger.info(
                f"[vapi_get_call_details] Waiting {delay} seconds before next attempt")
            time.sleep(delay)
            # Gradually increase delay but cap it at 30 seconds
            delay = min(delay * 1.5, 30.0)
        else:
            logger.warning(
                f"[vapi_get_call_details] Max retries reached. Returning partial data if available.")
            # Return whatever data we have even if incomplete
            if resp.status_code == 200:
                return data

    logger.error(
        f"[vapi_get_call_details] All retries failed for {vapi_call_id}")
    return None


def extract_summary_transcript_metadata(payload: dict):
    """
    Extract summary, transcript, metadata from a Vapi call payload (be tolerant to structure).
    Returns (summary, transcript, metadata)
    """
    if not payload:
        return None, None, {}

    logger.info(
        f"[extract_summary_transcript_metadata] Payload keys: {list(payload.keys())}")

    # Handle nested data structure
    data = payload.get("data", {}) if isinstance(
        payload.get("data", {}), dict) else {}

    # Extract summary from various possible locations
    summary = (
        payload.get("summary")
        or data.get("summary")
        or payload.get("analysis", {}).get("summary")
        or payload.get("analysis", {}).get("transcript_summary")
        or data.get("analysis", {}).get("summary")
        or data.get("analysis", {}).get("transcript_summary")
        or payload.get("artifact", {}).get("summary")
    )

    # Extract transcript from various possible locations
    transcript = None

    # Check direct transcript field
    if payload.get("transcript"):
        transcript = payload["transcript"]
    # Check in data.transcript
    elif data.get("transcript"):
        transcript = data["transcript"]
    # Check in messages array - format it properly
    elif payload.get("messages") and isinstance(payload["messages"], list):
        messages_transcript = []
        for msg in payload["messages"]:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content") or msg.get(
                    "message") or msg.get("text")
                if content:
                    messages_transcript.append(f"{role}: {content}")
        if messages_transcript:
            transcript = "\n".join(messages_transcript)
    # Check in recording.transcript
    elif payload.get("recording", {}).get("transcript"):
        transcript = payload["recording"]["transcript"]
    # Check in data.recording.transcript
    elif data.get("recording", {}).get("transcript"):
        transcript = data["recording"]["transcript"]
    # Check in artifact
    elif payload.get("artifact", {}).get("transcript"):
        transcript = payload["artifact"]["transcript"]
    # Check in analysis.transcript
    elif payload.get("analysis", {}).get("transcript"):
        transcript = payload["analysis"]["transcript"]

    metadata = payload.get("metadata") or data.get("metadata") or {}

    logger.info(
        f"[extract_summary_transcript_metadata] Extracted - Summary: {'Yes' if summary else 'No'}, "
        f"Transcript length: {len(transcript) if transcript else 0}, "
        f"Metadata keys: {list(metadata.keys()) if metadata else 'None'}"
    )

    return summary, transcript, metadata


# Global dicts/mappings
email_and_transcript = {}
email_and_transcript_lock = threading.Lock()

call_id_to_sid = {}  # Maps call_id (queue id) -> Twilio call SID
call_id_to_sid_lock = threading.Lock()

# Vapi mappings for fallback usage
call_id_to_vapi_id = {}  # Maps queue call_id -> Vapi call ID
call_id_to_vapi_id_lock = threading.Lock()

call_sid_to_vapi_id = {}  # Maps Twilio SID -> Vapi call ID
call_sid_to_vapi_id_lock = threading.Lock()

# contains call identifiers (Twilio SID or Vapi ID) already posted to SF
posted_call_ids = set()
posted_call_ids_lock = threading.Lock()

init_db(logger=logger)  # Initialize the database at startup

TERMINAL_STATUSES = {
    "completed",
    "busy",
    "failed",
    "no-answer",
    "cancelled",
    "canceled",
    "failed",
    "busy"
}
PROMPT_FILE = "prompts.txt"


class LoginRequest(BaseModel):
    email: str
    password: str

# --- Phone Number Formatting and Validation ---


def format_and_validate_number(raw_number, country=None):
    if not raw_number or not str(raw_number).strip():
        logger.warning(
            f"[format_and_validate_number] Empty or invalid raw_number: '{raw_number}' (country: '{country}')\n\n")
        return None

    raw_number = str(raw_number).strip()
    cleaned_number = re.sub(r"[^\d+]", "", raw_number)
    logger.info(
        f"[format_and_validate_number] Cleaned number: '{cleaned_number}' from raw input: '{raw_number}'\n\n")

    # If phone number starts with '+', assume country code is present and dial as is
    if cleaned_number.startswith("+"):
        logger.info(
            f"[format_and_validate_number] Number '{cleaned_number}' already has country code. Using as is.\n\n")
        return cleaned_number

    # If not, prepend country code from country_code column (no '+' enforced)
    country_code = None
    if country:
        country_code = str(country).strip()
        # Remove any non-digit characters and ensure it starts with +
        country_code = re.sub(r"\D", "", country_code)
        if not country_code.startswith("+"):
            country_code = "+" + country_code
    else:
        country_code = "+1"  # Default to US if not provided

    final_number = country_code + cleaned_number
    logger.info(
        f"[format_and_validate_number] Prepending country code. Final number: '{final_number}'\n\n")
    return final_number


class PromptRequest(BaseModel):
    prompt: str


async def verify_credentials(email: str, password: str) -> bool:
    """Verify credentials against Appwrite database"""
    try:
        if not validate_email(email):
            return False

        # Query Appwrite database for user with matching email
        result = databases.list_documents(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_COLLECTION_ID,
            queries=[Query.equal("email", email)]
        )

        if not result['documents']:
            logger.info(
                f"[verify_credentials] No user found with email: {email}")
            return False

        user = result['documents'][0]
        stored_password = user['password']

        # Direct password comparison (consider using hashed passwords in production)
        if stored_password == password:
            logger.info(
                f"[verify_credentials] Authentication successful for: {email}")
            return True
        else:
            logger.info(
                f"[verify_credentials] Authentication failed for: {email}")
            return False

    except AppwriteException as e:
        logger.error(f"[verify_credentials] Appwrite error: {e}")
        return False
    except Exception as e:
        logger.error(f"[verify_credentials] Unexpected error: {e}")
        return False


def create_session_token():
    """Create a new session token"""
    return secrets.token_urlsafe(32)


def get_current_user(
    session_token: Optional[str] = Cookie(None),
    session_token_query: Optional[str] = None  # Add this for Swagger
):
    # Use query parameter if cookie is not available
    token = session_token or session_token_query
    if not token or token not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return active_sessions[token]


def validate_email(email: str) -> bool:
    """Validate email format"""
    email_pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(email_pattern.match(email))


def poll_twilio_status(call_sid, call_id, customer_id, customer_name, max_wait=1200, poll_interval=5):
    """
    Poll Twilio for call status. If the call is completed or fails, remove it from the queue and process the next.
    """
    logger.info(
        f"[poll_twilio_status] Starting polling for callSid: {call_sid} (call_id: {call_id}, customer_id: {customer_id})\n\n")

    account_sid = getattr(settings, "TWILIO_ACCOUNT_SID", None)
    auth_token = getattr(settings, "TWILIO_AUTH_TOKEN", None)

    if not account_sid or not auth_token:
        logger.error(
            "[poll_twilio_status] Missing Twilio credentials in settings.\n\n")
        return

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json"
    auth = HTTPBasicAuth(account_sid, auth_token)

    elapsed = 0
    last_status = None

    while elapsed < max_wait:
        try:
            response = requests.get(url, auth=auth, timeout=10)
            if response.status_code == 200:
                data = response.json()
                current_status = data.get("status")
                logger.info(
                    f"[poll_twilio_status] Twilio callSid {call_sid} status: {current_status}\n\n")

                # UPDATE STATUS IMMEDIATELY WHEN IT CHANGES
                if current_status != last_status:
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute(
                            "UPDATE customer_data SET last_call_status = ? WHERE call_id = ?",
                            (current_status, call_id)
                        )
                        conn.commit()
                        conn.close()
                        logger.info(
                            f"[poll_twilio_status] Updated last_call_status to '{current_status}' for call_id {call_id}")
                        last_status = current_status
                    except Exception as update_error:
                        logger.error(
                            f"[poll_twilio_status] Failed to update last_call_status: {update_error}")

                if current_status in TERMINAL_STATUSES:
                    logger.info(
                        f"[poll_twilio_status] Terminal status '{current_status}' received for callSid {call_sid}.\n\n")

                    # === FIX: Update notes for ALL terminal statuses, not just "completed" ===
                    if current_status == "completed":
                        # Process transcript for completed calls
                        vapi_call_id = None
                        with call_sid_to_vapi_id_lock:
                            vapi_call_id = call_sid_to_vapi_id.get(call_sid)

                        if vapi_call_id:
                            vapi_data = vapi_get_call_details(vapi_call_id)
                            if vapi_data:
                                logger.info(
                                    f"[poll_twilio_status] Retrieved Vapi data for call_id {call_id} using vapi_call_id {vapi_call_id}")
                                summary, transcript, metadata = extract_summary_transcript_metadata(
                                    vapi_data)
                                if summary:
                                    logger.info(
                                        f"[poll_twilio_status] Processing transcript for call_id {call_id}")
                                    try:
                                        parsed = summarize_conversation_transcript(
                                            summary)
                                        logger.info(
                                            f"[poll_twilio_status] Parsed summary and tasks for call_id {call_id}: {parsed}")
                                        update_customer_data_notes_and_tasks(
                                            call_id=call_id,
                                            parsed=parsed,
                                            db_path=DB_PATH
                                        )
                                    except Exception as process_exc:
                                        logger.error(
                                            f"[poll_twilio_status] Error processing transcript: {process_exc}")
                                    # GET CUSTOMER EMAIL FROM CUSTOMER DATA TABLE USING THE CALL ID
                                    customer_email = None
                                    try:
                                        conn = sqlite3.connect(DB_PATH)
                                        c = conn.cursor()
                                        c.execute(
                                            "SELECT email FROM customer_data WHERE call_id = ?", (call_id,))
                                        result = c.fetchone()
                                        if result:
                                            customer_email = result[0]
                                        conn.close()
                                    except Exception as email_exc:
                                        logger.error(
                                            f"[poll_twilio_status] Error fetching customer email: {email_exc}")
                                    try:
                                        send_meeting_invite(
                                            parsed, customer_name, customer_email)
                                    except Exception as send_exc:
                                        logger.error(
                                            f"[poll_twilio_status] Error sending meeting invite: {send_exc}")
                    else:
                        # For non-completed calls (busy, no-answer, failed), create appropriate notes
                        status_notes = {
                            "busy": "Call ended - recipient's line was busy",
                            "no-answer": "Call ended - no answer from recipient",
                            "failed": "Call failed to connect",
                            "cancelled": "Call was cancelled"
                        }

                        note = status_notes.get(
                            current_status, f"Call ended with status: {current_status}")
                        logger.info(
                            f"[poll_twilio_status] Updating notes for {current_status} status: {note}")

                        update_customer_data_notes_and_tasks(
                            call_id=call_id,
                            parsed={"notes": note},
                            db_path=DB_PATH
                        )

                    # Remove from queue for all terminal statuses
                    logger.info(
                        f"removing {call_id} from queue after terminal status {current_status}\n\n")
                    pop_call_by_id(call_id)
                    threading.Thread(
                        target=process_queue_single_run, daemon=True).start()
                    return
            elif response.status_code == 404:
                logger.error(
                    f"[poll_twilio_status] Twilio callSid {call_sid} not found (404). Removing from queue.\n\n")
                # Update status to "not-found" before removing
                try:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute(
                        "UPDATE customer_data SET last_call_status = ? WHERE call_id = ?",
                        ("not-found", call_id)
                    )
                    conn.commit()
                    conn.close()
                except Exception as update_error:
                    logger.error(
                        f"[poll_twilio_status] Failed to update not-found status: {update_error}")

                pop_call_by_id(call_id)
                threading.Thread(
                    target=process_queue_single_run, daemon=True).start()
                return

        except Exception as e:
            logger.error(
                f"[poll_twilio_status] Exception while polling Twilio: {e}\n\n")

        time.sleep(poll_interval)
        elapsed += poll_interval

    # Handle timeout case
    logger.info(
        f"[poll_twilio_status] Max wait time exceeded for callSid {call_sid}. No terminal status received.\n\n")

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "UPDATE customer_data SET last_call_status = ? WHERE call_id = ?",
            ("timeout", call_id)
        )
        conn.commit()
        conn.close()
        logger.info(
            f"[poll_twilio_status] Updated last_call_status to 'timeout' for call_id {call_id}")
    except Exception as update_error:
        logger.error(
            f"[poll_twilio_status] Failed to update timeout status: {update_error}")

    # Remove timed out call from queue
    pop_call_by_id(call_id)
    threading.Thread(target=process_queue_single_run, daemon=True).start()


def initiate_call(
    phone_number: str,
    customer_details: str,  # Rename from 'details' to 'customer_details'
    llm_prompt: str,        # Add new parameter for LLM prompt
    lead_name: str,
    customer_id: str,
    correlation_id: str,
    call_id: Optional[int] = None,
    email: Optional[str] = None,
    country_code: Optional[str] = None
) -> bool:
    logger.info(f"[initiate_call] country code: {country_code}")
    logger.info(f"[initiate_call] phone number: {phone_number}")

    # Function to update status in database
    def update_call_status(status, error_message=None):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "UPDATE customer_data SET last_call_status = ? WHERE call_id = ?",
                (status, call_id)
            )
            conn.commit()
            conn.close()
            logger.info(
                f"[{correlation_id}] Updated last_call_status to '{status}' for call_id {call_id}")

            # Also update notes with error message if provided
            if error_message:
                update_customer_data_notes_and_tasks(
                    call_id=call_id,
                    parsed={"notes": f"Call initiation failed: {error_message}"},
                    db_path=DB_PATH
                )
        except Exception as update_error:
            logger.error(
                f"[{correlation_id}] Failed to update status: {update_error}")

    try:
        logger.info(
            f"[initiate_call] Starting outbound call for {lead_name} (SF ID: {customer_id})\n\n")

        # Format phone number with country code in E.164 format
        phone_number_final = format_and_validate_number(
            phone_number, country_code)
        if not phone_number_final:
            error_msg = f"Invalid phone number format: {phone_number} with country code: {country_code}"
            logger.error(f"[{correlation_id}] {error_msg}")
            update_call_status("invalid-phone", error_msg)
            return False

        logger.info(
            f"[initiate_call] Using phone number: {phone_number_final}\n\n")
        logger.info(
            f"[{correlation_id}] Initiating outbound call to {phone_number_final} with email: {email} being sent to initiate call function.\n\n")

        # Build dynamic variables for Vapi call
        # Build dynamic variables for Vapi call
        # Build dynamic variables for Vapi call
        dynamic_vars = {
            "first_message": generate_initial_message(customer_details),
            "customer_name": lead_name,
            "customer_details": customer_details,  # Customer-facing details
            "llm_prompt": llm_prompt,              # LLM prompt as separate variable
            "customer_id": customer_id,
            "email": email or "Please check the details for email",
            "call_id": call_id,
            "today_date": datetime.now().strftime("%Y-%m-%d")
        }

        # Or if you don't want to send LLM prompt to Vapi at all, remove it completely

        # Create assistant overrides
        assistant_overrides = {
            "firstMessage": dynamic_vars["first_message"],
            "variableValues": {
                "customer_name": lead_name,
                "customer_details": customer_details,  # Only customer details
                "email": email or "Please check the details for email",
                "today_date": datetime.now().strftime("%Y-%m-%d")
            }
        }

        # Make the outbound call via Vapi
        result = vapi_start_outbound_call(
            to_number=phone_number_final,
            metadata=dynamic_vars,
            assistant_overrides=assistant_overrides
        )

        logger.info(
            f"[{correlation_id}] Outbound call API result: {result}\n\n")

        # Parse returned identifiers
        vapi_call_id = (
            result.get("id")
            or result.get("data", {}).get("id")
        )
        call_sid = (
            result.get("twilioCallSid")
            or result.get("callSid")
            or result.get("data", {}).get("twilioCallSid")
            or result.get("data", {}).get("callSid")
            or result.get("transport", {}).get("callSid")
        )

        # Save email to global dict keyed by best-available id
        call_key_for_email = call_sid or vapi_call_id
        if call_key_for_email:
            with email_and_transcript_lock:
                email_and_transcript[str(call_key_for_email)] = {
                    "email": email,
                    "transcript": None
                }

        # Store mapping for fallback from Twilio->Vapi and queue_id->Vapi
        if call_id and vapi_call_id:
            with call_id_to_vapi_id_lock:
                call_id_to_vapi_id[call_id] = vapi_call_id
                logger.info(
                    f"[initiate_call] Mapped queue call_id {call_id} to vapi_call_id {vapi_call_id}")
        if call_sid and vapi_call_id:
            with call_sid_to_vapi_id_lock:
                call_sid_to_vapi_id[call_sid] = vapi_call_id
                logger.info(
                    f"[initiate_call] Mapped Twilio call_sid {call_sid} to vapi_call_id {vapi_call_id}")

        # Start background polling for Twilio status if Twilio SID present
        if call_sid and call_id is not None:
            with call_id_to_sid_lock:
                call_id_to_sid[str(call_id)] = str(call_sid)

            threading.Thread(
                target=poll_twilio_status,
                args=(call_sid, call_id, customer_id, lead_name),
                daemon=True
            ).start()
        else:
            logger.info(
                f"[{correlation_id}] No Twilio call_sid available for polling")

        logger.info(
            f"[{correlation_id}] Successfully initiated call to {phone_number_final} (Customer ID: {customer_id})\n\n")

        # Update status to "initiated" when call starts successfully
        update_call_status("initiated")
        return True

    except requests.HTTPError as http_err:
        # Handle specific HTTP errors
        error_message = str(http_err)
        status_code = http_err.response.status_code if hasattr(
            http_err, 'response') else 'unknown'

        logger.error(
            f"[{correlation_id}] HTTP Error {status_code}: {error_message}")

        if status_code == 400:
            if "valid phone number" in error_message or "E.164" in error_message:
                update_call_status("invalid-phone", error_message)
            else:
                update_call_status("initialize-failed", error_message)
        elif status_code == 401:
            update_call_status("auth-error", error_message)
        elif status_code == 404:
            update_call_status("not-found", error_message)
        elif status_code == 429:
            update_call_status("rate-limit", error_message)
        elif status_code >= 500:
            update_call_status("server-error", error_message)
        else:
            update_call_status("api-error", error_message)

        return False

    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        logger.error(
            f"[{correlation_id}] Call failed. Error while making the call: {error_message}\n\n")
        update_call_status("call-failed", error_message)
        return False


def load_all_prompts() -> str:
    """Reads the current prompt from the prompt file (only one prompt allowed)."""
    if not os.path.exists(PROMPT_FILE):
        return ""
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        # Return only the first line (current prompt) or empty string
        lines = [line.strip() for line in f if line.strip()]
        return lines[0] if lines else ""
# --- Shared Queue Processing Function ---


def process_queue_single_run():
    """Process the next queued call with enhanced error handling and status tracking"""
    logger.info("[process_queue_single_run] Checking queue for next call.")
    conn = None

    try:
        # First, check if we already have a processing call
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM call_queue WHERE status = 'processing'")
        if c.fetchone()[0] > 0:
            logger.info(
                "[process_queue_single_run] A call is already in processing. Exiting.")
            return
    except Exception as db_exc:
        logger.error(
            f"[process_queue_single_run] DB error while checking processing count: {db_exc}", exc_info=True)
        return
    finally:
        if conn:
            conn.close()

    # Handle log rotation in a separate function
    try:
        rotate_logs_if_needed()
    except Exception as log_exc:
        logger.warning(
            f"[process_queue_single_run] Log rotation failed: {log_exc}")

    try:
        # Get next queued call
        next_call = pop_next_call()
        if not next_call:
            logger.info("[process_queue_single_run] No queued calls found.")
            try:
                export_customer_data_to_excel(
                    db_path=DB_PATH, excel_path="resultant_excel.xlsx")
            except Exception as export_exc:
                logger.error(
                    f"[process_queue_single_run] Excel export error: {export_exc}")
            return

        # Extract call details
        call_id, customer_name, customer_id, phone_number, email, customer_requirements, notes, tasks, specific_prompt = next_call

        # Get additional customer data including specific_prompt
        customer_data = None
        specific_prompt = ""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "SELECT company_name, country_code, industry, location, specific_prompt FROM customer_data WHERE call_id = ?", (call_id,))
                customer_data = c.fetchone()

            if customer_data:
                company_name, country_code, industry, location, specific_prompt = customer_data
                company_name = company_name.strip() if company_name else None
                country_code = country_code.strip() if country_code else None
                industry = industry.strip() if industry else None
                location = location.strip() if location else None
                specific_prompt = specific_prompt.strip() if specific_prompt else ""

                logger.info(
                    f"[process_queue_single_run] Additional data retrieved for call_id {call_id}:")
                logger.info(f"  Company: {company_name}")
                logger.info(f"  Country Code: {country_code}")
                logger.info(f"  Industry: {industry}")
                logger.info(f"  Location: {location}")
                logger.info(f"  Specific Prompt: {specific_prompt}")
            else:
                logger.warning(
                    f"[process_queue_single_run] No additional customer data found for call_id: {call_id}")
                company_name = country_code = industry = location = specific_prompt = None

        except Exception as data_exc:
            logger.error(
                f"[process_queue_single_run] Error getting customer data: {data_exc}")
            company_name = country_code = industry = location = specific_prompt = None

        # Build call details
        # Build call details
        current_prompt = load_all_prompts()
        customer_specific_prompt = specific_prompt.strip() if specific_prompt else ""

        # If both specific_prompt and current_prompt are available, combine them.
        if customer_specific_prompt and current_prompt:
            final_prompt = f"this is customer specific prompt:{customer_specific_prompt}\n\nthis is the general prompt:{current_prompt}"
        elif customer_specific_prompt:
            final_prompt = f"this is customer specific prompt:{customer_specific_prompt}"
        else:
            final_prompt = f"this is the general prompt:{current_prompt}"

        # Handle notes and greeting
        if notes:
            llm_prompt = f"{final_prompt}\n\nCustomer Notes:\n{notes}"
        else:
            llm_prompt = f"{final_prompt}\n\n."

        # Build customer details (without LLM prompt)
        customer_details = f"These are the details of the customer you are speaking with. Name: {customer_name}:\n\n"
        customer_details += f"Customer Requirements: {customer_requirements}\n"
        customer_details += f"Notes: {notes}\n"
        customer_details += f"Company Name: {company_name}\n"
        customer_details += f"Country Code: {country_code}\n"
        customer_details += f"Industry: {industry}\n"
        customer_details += f"Location: {location}\n"

        # Log the processing
        logger.info(
            f"[process_queue_single_run] Processing call for {customer_name} (ID: {customer_id})")

        # Validate phone number
        phone = phone_number.strip() if phone_number else ""
        if not phone:
            logger.error(
                f"[process_queue_single_run] Invalid phone number for call_id {call_id}")
            pop_call_by_id(call_id)
            threading.Thread(target=process_queue_single_run,
                             daemon=True).start()
            return

        # Try to initiate the call
        correlation_id = str(uuid.uuid4())
        try:
            call_success = initiate_call(
                phone_number=phone,
                customer_details=customer_details,  # Customer details only
                llm_prompt=llm_prompt,              # LLM prompt separately
                lead_name=customer_name,
                customer_id=customer_id,
                correlation_id=correlation_id,
                call_id=call_id,
                email=email,
                country_code=country_code
            )

            if not call_success:
                logger.error(
                    f"[process_queue_single_run] Call initiation failed for call_id {call_id}")
                # Status is already updated by initiate_call function, just remove from queue
                pop_call_by_id(call_id)
                threading.Thread(
                    target=process_queue_single_run, daemon=True).start()
            else:
                logger.info(
                    f"[process_queue_single_run] Call successfully initiated for call_id {call_id}")

        except Exception as call_exc:
            logger.error(
                f"[process_queue_single_run] Call initiation error for call_id {call_id}: {call_exc}")
            # Update status for unexpected errors during initiation
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    c = conn.cursor()
                    c.execute(
                        "UPDATE customer_data SET last_call_status = ? WHERE call_id = ?",
                        ("initiation-error", call_id)
                    )
            except Exception as update_exc:
                logger.error(
                    f"[process_queue_single_run] Failed to update error status: {update_exc}")

            pop_call_by_id(call_id)
            threading.Thread(target=process_queue_single_run,
                             daemon=True).start()

    except Exception as e:
        logger.error(
            f"[process_queue_single_run] Fatal error: {e}", exc_info=True)
        if 'call_id' in locals():
            try:
                pop_call_by_id(call_id)
            except Exception:
                pass
        threading.Thread(target=process_queue_single_run, daemon=True).start()


def rotate_logs_if_needed():
    """Handle log rotation when file gets too large"""
    try:
        if os.path.exists("app.log") and os.path.getsize("app.log") > 1024 * 1024:  # 1MB
            with open("app.log", "r") as f:
                lines = f.readlines()
            if len(lines) > 1000:
                with open("app.log", "w") as f:
                    f.writelines(lines[-1000:])  # Keep last 1000 lines
                logger.info(
                    f"Rotated log file, kept last 1000 lines from {len(lines)} total")
    except Exception as e:
        logger.error(f"Error rotating log file: {e}", exc_info=True)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    logger.info("[root API] / endpoint called. Returning HTML dashboard.\n\n")
    return templates.TemplateResponse(
        "Index.html",
        {"request": request, "message": "Welcome to the Call Queue API 🚀"}
    )


@app.post("/login")
async def login(data: LoginRequest):
    """Login endpoint that validates email format and sets cookie"""
    if not await verify_credentials(data.email, data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    session_token = create_session_token()
    active_sessions[session_token] = data.email

    response = JSONResponse(content={"message": "Login successful"})
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        max_age=60*60*24*30,    # 30 days
        samesite="lax"
    )
    return response


@app.post("/logout")
async def logout(session_token: Optional[str] = Cookie(None)):
    if session_token and session_token in active_sessions:
        del active_sessions[session_token]

    response = JSONResponse(content={"message": "Logout successful"})
    response.delete_cookie(key="session_token")
    return response


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request, username: str = Depends(get_current_user)):
    logger.info(f"[upload_page API] User {username} accessing upload page\n\n")
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/submit-prompt")
async def submit_prompt(
    payload: PromptRequest,
    user: str = Depends(get_current_user)
):
    prompt_text = payload.prompt

    if not prompt_text:
        raise HTTPException(status_code=400, detail="Prompt text is required")

    # OVERWRITE the prompt file with the new prompt (instead of appending)
    with open(PROMPT_FILE, "w", encoding="utf-8") as f:  # Changed "a" to "w"
        f.write(prompt_text.strip() + "\n")  # Only write the current prompt

    return {"status": "success", "message": "Prompt saved successfully (overwritten)"}


@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...), username: str = Depends(get_current_user)):
    logger.info(
        f"[upload_file API] User {username} uploading file: {file.filename}\n\n")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM customer_data")
    logger.info("Deleted previous customer data.")
    # c.execute("DELETE FROM call_queue")
    # logger.info("Cleared previous call queue.")
    conn.commit()
    conn.close()
    TEMP_FILE_PATH = "temp_upload.xlsx"
    try:
        with open(TEMP_FILE_PATH, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        return JSONResponse(content={"message": "File uploaded successfully"})
    except Exception as e:
        return JSONResponse(content={"message": f"Upload failed: {str(e)}"}, status_code=500)


@app.get("/auth-check")
async def auth_check(username: str = Depends(get_current_user)):
    return {"authenticated": True, "username": username}

# Endpoint to download the resultant Excel file


@app.get("/download-excel")
def download_excel(username: str = Depends(get_current_user)):
    """
    Delivers the resultant Excel file as a downloadable response.
    """
    logger.info(f"[download-excel] User {username} downloading Excel file\n\n")
    # Always export the latest customer_data to Excel before serving
    excel_path = os.path.join(os.getcwd(), "resultant_excel.xlsx")
    try:
        export_customer_data_to_excel(db_path=DB_PATH, excel_path=excel_path)
    except Exception as e:
        logger.error(f"[download_excel] Error exporting Excel: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to generate Excel file.")
    if not os.path.exists(excel_path):
        logger.error(f"[download_excel] File not found: {excel_path}")
        raise HTTPException(status_code=404, detail="Excel file not found.")
    return FileResponse(
        path=excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="resultant_excel.xlsx"
    )


@app.get("/excel-status")
def excel_status(username: str = Depends(get_current_user)):
    """
    Returns the resultant Excel file if it exists, else a 'file isn't ready yet' message.
    """
    logger.info(f"[excel-status] User {username} checking Excel status\n\n")

    excel_path = os.path.join(os.getcwd(), "resultant_excel.xlsx")
    if os.path.exists(excel_path):
        return FileResponse(
            path=excel_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename="resultant_excel.xlsx"
        )
    else:
        return {"message": "File isn't ready yet."}


@app.post("/add-call")
async def add_call(username: str = Depends(get_current_user)):
    """
    Processes the previously uploaded Excel file (temp_upload.xlsx) and adds calls to the queue.
    """
    logger.info(
        f"[add_call API] User {username} processing previously uploaded Excel file\n\n")
    TEMP_FILE_PATH = "temp_upload.xlsx"
    try:
        if not os.path.exists(TEMP_FILE_PATH):
            logger.error("No file uploaded yet. temp_upload.xlsx not found.")
            raise HTTPException(
                status_code=400, detail="No file uploaded yet. Please upload an Excel file first.")

        df = pd.read_excel(TEMP_FILE_PATH)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Clear customer_data table before adding new batch
        c.execute("DELETE FROM customer_data")
        logger.info(f"Deleted previous customer data.")
        # c.execute("DELETE FROM call_queue")
        # logger.info(f"Cleared previous call queue.")
        added_count = 0
        for _, row in df.iterrows():
            specific_prompt = row.get('specific_prompt', '')
            customer_name_raw = row.get('customer_name', '')
            if pd.isna(customer_name_raw):
                customer_name = None
            else:
                customer_name = str(customer_name_raw).strip()
            customer_id = (str(row.get('customer_id', '')) if row.get(
                'customer_id', '') is not None else '').strip()
            # Sanitize phone_number
            phone_number_raw = row.get('phone_number', '')
            if phone_number_raw is None or (isinstance(phone_number_raw, float) and math.isnan(phone_number_raw)):
                phone_number = ''
            elif isinstance(phone_number_raw, float):
                phone_number = str(int(phone_number_raw))
            else:
                phone_number = str(phone_number_raw).strip()
            # Sanitize country_code
            country_code_raw = row.get('country_code', '')
            if country_code_raw is None or (isinstance(country_code_raw, float) and math.isnan(country_code_raw)):
                country_code = ''
            elif isinstance(country_code_raw, float):
                country_code = str(int(country_code_raw))
            else:
                country_code = str(country_code_raw).strip()
            logger.info(f"country_code: {country_code}")
            email = (str(row.get('email', '')) if row.get(
                'email', '') is not None else '').strip()
            customer_requirements = (str(row.get('customer_requirements', '')) if row.get(
                'customer_requirements', '') is not None else '').strip()

            def safe_str(value, default=''):
                if pd.isna(value) or value is None:
                    return default
                return str(value).strip()

            notes = safe_str(row.get('notes', ''))
            tasks = safe_str(row.get('tasks', ''))
            to_call = (str(row.get('to_call', '')) if row.get(
                'to_call', '') is not None else '').strip()
            industry = (str(row.get('industry', '')) if row.get(
                'industry', '') is not None else '').strip()
            company_name = (str(row.get('company_name', '')) if row.get(
                'company_name', '') is not None else '').strip()
            location = (str(row.get('location', '')) if row.get(
                'location', '') is not None else '').strip()
            logger.info(
                f"Processing row: specific_prompt={specific_prompt}, customer_id={customer_id}, customer_name={customer_name}, phone_number={phone_number}, to_call={to_call}")
            if to_call.lower() == "yes":
                # Insert into call_queue first to get call_id
                if customer_name and phone_number:
                    try:
                        c.execute("INSERT INTO call_queue (customer_name, customer_id, phone_number, email, customer_requirements, specific_prompt, to_call, notes, tasks, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued')", (
                            customer_name, customer_id, phone_number, email, customer_requirements, specific_prompt, to_call, notes, tasks))
                        call_id = c.lastrowid
                        # Insert into  customer_data with the same call_id
                        c.execute("""
                            INSERT OR REPLACE INTO customer_data (call_id, customer_name, customer_id, phone_number, email, customer_requirements, to_call, notes, tasks, country_code, industry, company_name, location, specific_prompt)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (call_id, customer_name, customer_id, phone_number, email, customer_requirements, to_call, notes, tasks, country_code, industry, company_name, location, specific_prompt))
                        added_count += 1
                    except Exception as e:
                        logger.error(f"[add_call] Failed to insert row: {e}")
        conn.commit()
        conn.close()
        response = {
            "message": f"Processed {len(df)} rows. Added {added_count} new entries to queue."
        }
        threading.Thread(target=process_queue_single_run, daemon=True).start()
        return response
    except Exception as e:
        logger.error(f"[add_call] Error processing Excel file: {e}")
        raise HTTPException(
            status_code=400, detail="Failed to process Excel file.")


@app.post("/webhook/call-ended")
async def call_ended(request: Request):
    logger.info("[call_ended API] Received call end webhook.\n\n")

    try:
        data = await request.json()
        logger.info(f"[call_ended] Webhook received data.")

        # Log all top-level keys and important subfields for debugging
        logger.info(f"[call_ended] Webhook received keys: {list(data.keys())}")
        logger.info(
            f"[call_ended] data['data'] keys: {list(data.get('data', {}).keys())}")
        logger.info(
            f"[call_ended] dynamic_variables: {list(data.get('data', {}).get('conversation_initiation_client_data', {}).get('dynamic_variables', {}).keys())}")
        logger.info(
            f"[call_ended] analysis keys: {list(data.get('data', {}).get('analysis', {}).keys())}")
        logger.info(
            f"[call_ended] metadata keys: {list(data.get('data', {}).get('metadata', {}).keys())}")

        # Extract all possible variables from Vapi payload
        data_dict = data.get("data", {}) if isinstance(
            data.get("data", {}), dict) else {}
        metadata = data.get("metadata", {}) or data_dict.get(
            "metadata", {}) or {}
        dynamic_vars = {}

        # Try to get variables from conversation_initiation_client_data
        if isinstance(data_dict.get("conversation_initiation_client_data", {}), dict):
            dynamic_vars.update(data_dict.get(
                "conversation_initiation_client_data", {}).get("dynamic_variables", {}))

        # Also check direct metadata and any additional locations
        dynamic_vars.update(metadata if isinstance(metadata, dict) else {})

        # Get transcript from various possible locations
        call_transcript = (
            data.get("transcript")
            or data_dict.get("transcript")
            or (data.get("data", {}) or {}).get("transcript")
            or (data.get("analysis", {}) or {}).get("transcript")
            or (data.get("summary", {}) or {}).get("transcript")
            or (data.get("messages") if isinstance(data.get("messages"), str) else None)
            or (data.get("conversation") if isinstance(data.get("conversation"), str) else None)
        )

        # Extract critical fields
        call_id = dynamic_vars.get("call_id")
        customer_id = dynamic_vars.get("customer_id")
        customer_name = dynamic_vars.get("customer_name", "Unknown Customer")
        customer_email = dynamic_vars.get("email", "No email provided")

        # Get call_sid from various locations
        call_sid = (
            data.get("twilioCallSid")
            or data.get("twilio_call_sid")
            or data.get("callSid")
            or data.get("phoneCallProviderId")
            or (data.get("data", {}) or {}).get("twilioCallSid")
            or (data.get("transport", {}) or {}).get("callSid")
            or (data.get("data", {}).get("metadata", {}).get("phone_call", {}) or {}).get("call_sid")
        )

        # Get summary/analysis
        analysis = data_dict.get("analysis", {})
        call_summary = (
            analysis.get("transcript_summary")
            or data.get("summary")
            or "No summary provided"
        )

        logger.info(
            f"[call_ended] Extracted fields: call_sid={call_sid}, customer_id={customer_id}, customer_name={customer_name}, call_summary={'present' if call_summary else 'missing'}, call_transcript={'present' if call_transcript else 'missing'}")
        logger.info(f"[call_ended] Dynamic vars: {dynamic_vars}")
        logger.info(f"[call_ended] Metadata: {metadata}")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE customer_data SET last_call_status = ? WHERE call_id = ?",
                  ("completed", call_id))
        conn.commit()
        conn.close()

        # Save transcript to global email_and_transcript dict
        with email_and_transcript_lock:
            if str(call_sid) not in email_and_transcript:
                email_and_transcript[str(call_sid)] = {
                    "email": customer_email, "transcript": call_transcript}
            else:
                email_and_transcript[str(
                    call_sid)]["transcript"] = call_transcript

        if not customer_id:
            logger.error("Missing 'customer_id' in webhook payload.\n\n")
            raise HTTPException(
                status_code=400, detail="Missing customer_id in webhook.")

        # Validate required fields
        if not call_id and not customer_id:
            logger.error(
                "[call_ended] Missing both call_id and customer_id in webhook data")
            raise HTTPException(
                status_code=400,
                detail="Missing both call_id and customer_id in webhook data"
            )

        # Process transcript and update Excel
        try:
            parsed = None
            if call_transcript:
                try:
                    parsed = summarize_conversation_transcript(call_transcript)
                    logger.info("[call_ended] Successfully parsed transcript")
                except Exception as parse_exc:
                    logger.error(
                        f"[call_ended] Failed to parse transcript: {parse_exc}")
                    parsed = None  # Fallback to empty parse

            # Always update notes/tasks even if parsing fails
            try:
                update_customer_data_notes_and_tasks(
                    call_id=call_id,
                    parsed=parsed,
                    db_path=DB_PATH
                )
                logger.info(
                    f"[call_ended] Updated notes/tasks for call_id {call_id}")
            except Exception as update_exc:
                logger.error(
                    f"[call_ended] Failed to update notes/tasks: {update_exc}")

            # Try to send meeting invite if we have contact info
            if customer_email and customer_email != "No email provided":
                try:
                    send_meeting_invite(
                        parsed=parsed,
                        customer_name=customer_name,
                        customer_email=customer_email
                    )
                    logger.info(
                        f"[call_ended] Sent meeting invite to {customer_email}")
                except Exception as invite_exc:
                    logger.error(
                        f"[call_ended] Failed to send meeting invite: {invite_exc}")

        except Exception as process_exc:
            logger.error(
                f"[call_ended] Error processing call data: {process_exc}")
            # Don't re-raise, continue to queue cleanup

        # Remove completed call from queue with robust error handling
        queue_cleanup_success = False
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                if call_id:  # Try call_id first
                    pop_call_by_id(call_id)
                    logger.info(
                        f"[call_ended] Removed queue entry by call_id {call_id}")
                    queue_cleanup_success = True
                elif customer_id:  # Fallback to customer_id
                    cursor.execute(
                        "SELECT call_id FROM call_queue WHERE customer_id = ? AND status = 'processing'",
                        (customer_id,)
                    )
                    rows = cursor.fetchall()
                    for row in rows:
                        queue_id = row[0] if isinstance(
                            row, (tuple, list)) else row
                        pop_call_by_id(queue_id)
                        logger.info(
                            f"[call_ended] Removed queue entry {queue_id} for customer_id {customer_id}")
                        queue_cleanup_success = True

                if not queue_cleanup_success:
                    logger.warning(
                        f"[call_ended] No queue entries found to remove. IDs: call_id={call_id}, customer_id={customer_id}")

        except Exception as db_exc:
            logger.error(
                f"[call_ended] Database error during queue cleanup: {db_exc}", exc_info=True)

        # Always try to trigger next call, regardless of cleanup success
        try:
            logger.info("[call_ended] Triggering next call after webhook")
            threading.Thread(target=process_queue_single_run,
                             daemon=True).start()
        except Exception as thread_exc:
            logger.error(
                f"[call_ended] Failed to start next call process: {thread_exc}")

        # Handle stuck calls older than 11 minutes
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT call_id, customer_id, customer_name, called_at
                    FROM call_queue
                    WHERE status = 'processing' AND called_at <= datetime('now', '-11 minutes')
                    ORDER BY called_at ASC
                    LIMIT 1
                """)
                stuck = cursor.fetchone()
                if stuck:
                    stuck_call_id, customer_id, customer_name, _ = stuck
                    logger.warning(
                        f"Stuck call detected (ID {stuck_call_id}), removing from queue.\n\n")

                    pop_call_by_id(stuck_call_id)

                    parsed = summarize_conversation_transcript(call_transcript)
                    update_customer_data_notes_and_tasks(
                        call_id=stuck_call_id, parsed=parsed, db_path=DB_PATH)
                    send_meeting_invite(
                        parsed=parsed, customer_name=customer_name, customer_email=customer_email)

                    threading.Thread(
                        target=process_queue_single_run, daemon=True).start()
        except Exception as stuck_exc:
            logger.error(
                f"Error checking for stuck calls: {stuck_exc}\n\n", exc_info=True)

        return {"status": "Webhook processed, queue updated.", "entity_id_processed": customer_id}

    except Exception as e:
        logger.error(
            f"Fatal error in call-ended webhook: {e}\n\n", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/status")
def queue_status():
    logger.info(
        "[status API] /status endpoint called. Returns current queue status.\n\n")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("""
                SELECT call_id, customer_id, customer_name, phone_number, email, status, created_at
                FROM call_queue 
                ORDER BY call_id ASC
            """)
            queue = [
                {
                    "call_id": row[0],
                    "customer_id": row[1],
                    "customer_name": row[2],
                    "phone": row[3],
                    "email": row[4],
                    "status": row[5],
                    "created_at": row[6]
                }
                for row in cursor.fetchall()
            ]
        return {"queue": queue}
    except Exception as e:
        logger.error("Error in /status: %s\n\n", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to fetch queue status.")


@app.get("/customer-data-status")
def customer_data_status():
    logger.info(
        "[status API] /customer-data-status endpoint called. Returns current customer data status.\n\n")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("""
                SELECT call_id, customer_id, customer_name, phone_number, email, country_code, specific_prompt
                FROM customer_data
                ORDER BY call_id ASC
            """)
            customer_data = [
                {
                    "call_id": row[0],
                    "customer_id": row[1],
                    "customer_name": row[2],
                    "phone": row[3],
                    "email": row[4],
                    "country_code": row[5],
                    "specific_prompt": row[6]
                }
                for row in cursor.fetchall()
            ]
        return {"queue": queue}
    except Exception as e:
        logger.error("Error in /customer-data-status: %s\n\n",
                     e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to fetch queue status.")


@app.post("/update-queue")
def update_queue(req: QueueUpdateRequest):
    logger.info(
        f"[update_queue API] /update-queue endpoint called. Updates queue entry with id: {req.id}.\n\n")

    try:
        fields = {
            "status": req.status,
            "phone_number": req.phone_number,
            "lead_name": req.lead_name,
            "details": req.details
        }

        updates = []
        params = []

        for column, value in fields.items():
            if value is not None:
                updates.append(f"{column} = ?")
                params.append(value)

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update.")

        params.append(req.id)

        query = f"""
            UPDATE call_queue 
            SET {', '.join(updates)} 
            WHERE id = ?
        """

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(query, params)
            conn.commit()

        return {"message": "Queue updated successfully."}

    except Exception as e:
        logger.error("Error in /update-queue: %s\n\n", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update queue.")


@app.delete("/delete-queue/{queue_id}")
def delete_queue_item(queue_id: int):
    logger.info(
        f"[delete_queue_item API] /delete-queue/{queue_id} endpoint called. Deleting queue item {queue_id}.\n\n")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM call_queue WHERE call_id = ?", (queue_id,))
            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404, detail="Queue item not found")
            conn.commit()

        # Trigger background process to handle next call
        threading.Thread(target=process_queue_single_run, daemon=True).start()

        return {"message": f"Queue item {queue_id} deleted successfully."}

    except HTTPException:
        raise  # re-raise for FastAPI to handle properly
    except Exception as e:
        logger.error(
            f"Error in /delete-queue/{queue_id}: {e}\n\n", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to delete queue item.")


@app.get("/delete-all-queue")
def delete_all_queue():
    logger.info(
        "[delete_all_queue API] /delete-all-queue endpoint called. Deleting all queue items.\n\n")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM call_queue")
            conn.commit()

        return {"message": "All queue items deleted successfully."}

    except Exception as e:
        logger.error("Error in /delete-all-queue: %s\n\n", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to delete all queue items.")


@app.get("/delete-customer-data-queue")
def delete_customer_data_queue():
    logger.info(
        "[delete_customer_data_queue API] /delete-customer-data-queue endpoint called. Deleting all customer data.\n\n")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM customer_data")
            conn.commit()

        return {"message": "All customer data deleted successfully."}

    except Exception as e:
        logger.error(
            "Error in /delete-customer-data-queue: %s\n\n", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to delete all customer data.")


def cleanup_stuck_calls():
    logger.info(
        "[cleanup_stuck_calls] Background thread started. Periodically cleaning up stuck calls.\n\n")

    while True:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT call_id, customer_id, customer_name, created_at
                    FROM call_queue
                    WHERE status = 'processing' AND created_at <= datetime('now', '-15 minutes')
                    ORDER BY created_at ASC
                    LIMIT 1
                """)
                stuck = cursor.fetchone()

                if stuck:
                    call_id, customer_id, customer_name, called_at = stuck
                    logger.warning(
                        f"[{call_id}] No response for 15 minutes. Logging and retrying next call.\n\n")

                    pop_call_by_id(call_id)

                    owner_id = "Unknown"

                    # Attempt to fetch owner and lead/contact name
                    # try:
                    #     fetch_map = {
                    #         "lead": fetch_lead_details,
                    #         "opportunity": fetch_opportunity_details,
                    #         "contact": fetch_contact_details
                    #     }
                    #     if entity_type in fetch_map:
                    #         data = fetch_map[entity_type](entity_id)
                    #         owner_id = data.get("OwnerId", "Unknown")
        except Exception as detail_exc:
            #         logger.error(f"[{call_id}] Failed to fetch {entity_type} details: {detail_exc}")
            # except Exception as fetch_exc:
            #     logger.error(f"[{call_id}] Error fetching entity details: {fetch_exc}")

            # Log the timeout event in notes/tasks
            try:
                timeout_note = f"Call timed out after 15 minutes of no response. Owner: {owner_id}"
                update_customer_data_notes_and_tasks(
                    call_id=call_id,
                    parsed={"notes": timeout_note},
                    db_path=DB_PATH
                )
                logger.info(
                    f"[{call_id}] Logged timeout note to customer data.")
            except Exception as note_exc:
                logger.error(
                    f"[{call_id}] Failed to log timeout note: {note_exc}")

            threading.Thread(
                target=process_queue_single_run, daemon=True).start()
