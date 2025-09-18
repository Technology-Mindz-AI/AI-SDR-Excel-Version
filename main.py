import uuid
import sqlite3
import requests
import time
import threading
from typing import Optional
import re
from datetime import datetime
from elevenlabs import ElevenLabs
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

app = FastAPI(title="Call Queue")


active_sessions = {}

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


# ElevenLabs client setup
client = ElevenLabs(api_key=settings.ELEVENLABS_API)

# Global dict to store both email and transcript by call_sid
email_and_transcript = {}
email_and_transcript_lock = threading.Lock()

call_id_to_sid = {}  # Maps call_sid to call_id
call_id_to_sid_lock = threading.Lock()

init_db(logger=logger)  # Initialize the database at startup

TERMINAL_STATUSES = {"completed", "busy", "failed", "no-answer", "cancelled"}

PROMPT_FILE = "prompts.txt"

class LoginRequest(BaseModel):
    email: str
    password: str

# --- Phone Number Formatting and Validation ---
# def format_and_validate_number(raw_number, country=None):
#     if not raw_number or not str(raw_number).strip():
#         logger.warning(f"[format_and_validate_number] Empty or invalid raw_number: '{raw_number}' (country: '{country}')\n\n")
#         return None

#     raw_number = str(raw_number).strip()
    # cleaned_number = re.sub(r"[^\d+]", "", raw_number)
    # logger.info(f"[format_and_validate_number] Cleaned number: '{cleaned_number}' from raw input: '{raw_number}'\n\n")

    # # If phone number starts with '+', assume country code is present and dial as is
    # if cleaned_number.startswith("+"):
    #     logger.info(f"[format_and_validate_number] Number '{cleaned_number}' already has country code. Using as is.\n\n")
    #     return cleaned_number

    # # If not, prepend country code from country_code column (no '+' enforced)
    # country_code = None
    # if country:
    #     country_code = str(country).strip()
    #     # Remove any non-digit characters
    #     country_code = re.sub(r"\D", "", country_code)
    # else:
    #     country_code = "1"  # Default to US if not provided
    # final_number = country_code + cleaned_number
    # logger.info(f"[format_and_validate_number] Prepending country code. Final number: '{final_number}'\n\n")
    # return final_number

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
            logger.info(f"[verify_credentials] No user found with email: {email}")
            return False
            
        user = result['documents'][0]
        stored_password = user['password']
        
        # Direct password comparison (consider using hashed passwords in production)
        if stored_password == password:
            logger.info(f"[verify_credentials] Authentication successful for: {email}")
            return True
        else:
            logger.info(f"[verify_credentials] Authentication failed for: {email}")
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
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(email_pattern.match(email))

def poll_twilio_status(call_sid, call_id, customer_id, customer_name, max_wait=150, poll_interval=5):
    """
    Poll Twilio for call status. If the call is completed or fails, remove it from the queue and process the next.
    """
    logger.info(f"[poll_twilio_status] Starting polling for callSid: {call_sid} (call_id: {call_id}, customer_id: {customer_id})\n\n")

    account_sid = getattr(settings, "TWILIO_ACCOUNT_SID", None)
    auth_token = getattr(settings, "TWILIO_AUTH_TOKEN", None)

    if not account_sid or not auth_token:
        logger.error("[poll_twilio_status] Missing Twilio credentials in settings.\n\n")
        return

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json"
    auth = HTTPBasicAuth(account_sid, auth_token)

    elapsed = 0
    while elapsed < max_wait:
        try:
            response = requests.get(url, auth=auth, timeout=10)
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                logger.info(f"[poll_twilio_status] Twilio callSid {call_sid} status: {status}\n\n")

                if status in TERMINAL_STATUSES:
                    logger.info(f"[poll_twilio_status] Terminal status '{status}' received for callSid {call_sid}.\n\n")
                    logger.info(f"[poll_twilio_status] Updating customer_data with last_call_status: {status}\n\n")
                   
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("UPDATE customer_data SET last_call_status = ? WHERE call_id = ?", (status, call_id))
                    conn.commit()
                    conn.close()
                   
                    if status != "completed":

                        logger.warning(f"[poll_twilio_status] PARSED == NONE being passed to append_notes_and_tasks since call status: {status}")
                        update_customer_data_notes_and_tasks(call_id=call_id, parsed=None, db_path="queue.db")
                        logger.info(f"removing {call_id} from queue after terminal status {status}\n\n")
                        pop_call_by_id(call_id)
                    threading.Thread(target=process_queue_single_run, daemon=True).start()
                    return   
            elif response.status_code == 404:
                logger.error(f"[poll_twilio_status] Twilio callSid {call_sid} not found (404). Removing from queue.\n\n")
                pop_call_by_id(call_id)
                threading.Thread(target=process_queue_single_run, daemon=True).start()
                return   
            else:
                logger.warning(f"[poll_twilio_status] Twilio API error {response.status_code}: {response.text}\n\n")

        except Exception as e:
            logger.error(f"[poll_twilio_status] Exception while polling Twilio: {e}\n\n")

        time.sleep(poll_interval)
        elapsed += poll_interval

    logger.info(f"[poll_twilio_status] Max wait time exceeded for callSid {call_sid}. No terminal status received.\n\n")

def initiate_call(
    phone_number: str,
    details: str,
    lead_name: str,
    customer_id: str,
    correlation_id: str,
    call_id: Optional[int] = None,
    email: Optional[str] = None,
    country_code: Optional[str] = None
) -> bool:
    logger.info(f"[initiate_call] country code: {country_code}")
    logger.info(f"[initiate_call] phone number: {phone_number}")
    try:
        logger.info(f"[initiate_call] Starting outbound call for {lead_name} (SF ID: {customer_id})\n\n")
        phone_number_clean = (str(phone_number) if phone_number is not None else '').strip()
        # Sanitize country_code: remove decimals, whitespace, and ensure string
        country_code_clean = ''
        if country_code is not None and str(country_code).strip() != '':
            try:
                country_code_clean = str(int(float(country_code))).strip()
            except Exception:
                country_code_clean = str(country_code).strip()
        else:
            country_code_clean = '1'  # Default to US country code

        phone_number_final = country_code_clean + phone_number_clean
        
        logger.info(f"[initiate_call] Using phone number: {phone_number_final}\n\n")
        logger.info(f"[{correlation_id}] Initiating outbound call to {phone_number_final} with email: {email} being sent to initiate call function.\n\n")
        result = client.conversational_ai.twilio.outbound_call(
            agent_id=settings.AGENT_ID,
            agent_phone_number_id=settings.AGENT_PHONE_NUMBER_ID,
            to_number=phone_number_final,
            conversation_initiation_client_data={
                "dynamic_variables": {
                    "first_message": generate_initial_message(details),
                    "customer_id": customer_id,
                    "customer_name": lead_name,
                    "customer_details": details,
                    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "email": email or "Please check the details for email",
                    "call_id": call_id 
                }
            },
        )

        logger.info(f"[{correlation_id}] Outbound call API result: {result}\n\n")

        if hasattr(result, 'success') and result.success is False:
            logger.error(f"[{correlation_id}] Outbound call failed: {getattr(result, 'message', 'No message')}\n\n")
            return False

        # Start background polling for call status
        call_sid = getattr(result, 'callSid', None) or getattr(result, 'call_sid', None)
        if call_sid and call_id is not None:
            
            with call_id_to_sid_lock:
                call_id_to_sid[str(call_id)] = str(call_sid)
         
            # Save email to global email_and_transcript dict
            with email_and_transcript_lock:
                if str(call_sid) not in email_and_transcript:
                    email_and_transcript[str(call_sid)] = {"email": email, "transcript": None}
                else:
                    email_and_transcript[str(call_sid)]["email"] = email
            threading.Thread(
                target=poll_twilio_status,
                args=(call_sid, call_id, customer_id, lead_name),
                daemon=True
            ).start()

        logger.info(f"[{correlation_id}] Successfully initiated call to {phone_number_final} (Customer ID: {customer_id})\n\n")
        return True

    except Exception as e:
        logger.error(f"[{correlation_id}] Call failed. Error while making the call: {e}\n\n")
        update_customer_data_notes_and_tasks(call_id=call_id, parsed=None, db_path="queue.db")
        return False

def load_all_prompts() -> str:
    """Reads all saved prompts from the prompt file and concatenates them."""
    if not os.path.exists(PROMPT_FILE):
        return ""
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return "\n".join([line.strip() for line in f if line.strip()])

# --- Shared Queue Processing Function ---
def process_queue_single_run():
    logger.info("[process_queue_single_run] Checking queue for next call.\n\n")
    conn = None

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM call_queue WHERE status = 'processing'")
        if c.fetchone()[0] > 0:
            logger.info("[process_queue_single_run] A call is already in processing. Exiting.\n\n")
            return
    except Exception as db_exc:
        logger.error(f"[process_queue_single_run] DB error while checking processing count: {db_exc}\n\n", exc_info=True)
        return
    finally:
        if conn:
            conn.close()

    try:
        next_call = pop_next_call()
        
        # Optimized log rotation - only check/rotate periodically
        try:
            import os
            if os.path.exists("app.log") and os.path.getsize("app.log") > 1024 * 1024:  # Only check if file > 1MB
                with open("app.log", "r") as f:
                    lines = f.readlines()
                if len(lines) > 1000:
                    with open("app.log", "w") as f:
                        f.writelines(lines[-1000:])
                    logger.info(f"[process_queue_single_run] Rotated log file, kept last 1000 lines from {len(lines)} total")
        except Exception as log_exc:
            logger.warning(f"[process_queue_single_run] Log rotation failed: {log_exc}\n\n")

        if not next_call:
            logger.info("[process_queue_single_run] No queued calls found.\n\n")
            export_customer_data_to_excel(db_path="queue.db", excel_path="resultant_excel.xlsx")
            return

        call_id, customer_name, customer_id, phone_number, email, customer_requirements, notes, tasks = next_call
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT company_name, country_code, industry, location FROM customer_data WHERE call_id = ?", (call_id,))
        row = c.fetchone()
        if not row:
            logger.warning(f"[process_queue_single_run] No customer data found for call_id: {call_id}\n\n")
        if row:
            company_name, country_code, industry, location = row
            company_name = company_name.strip() if company_name else None
            logger.info(f"[process_queue_single_run] Company Name: {company_name}\n\n")
            country_code = country_code.strip() if country_code else None
            logger.info(f"[process_queue_single_run] Country Code: {country_code}\n\n")
            industry = industry.strip() if industry else None
            logger.info(f"[process_queue_single_run] Industry: {industry}\n\n")
            location = location.strip() if location else None
            logger.info(f"[process_queue_single_run] Location: {location}\n\n")
        else:
            company_name, country_code, industry, location = None, None, None, None
        conn.commit()
        conn.close()
        logger.info(f"[process_queue_single_run] Picked call_id: {call_id} for {customer_id}:{phone_number}\n\n")


        notes = notes.strip()
        logger.info(f"[process_queue_single_run] Notes: {notes}\n\n")
        logger.info(f"[process_queue_single_run] Customer Requirements: {customer_requirements}\n\n")
        logger.info(f"[process_queue_single_run] Customer ID: {customer_id}\n\n")
        logger.info(f"[process_queue_single_run] Creating the details for dynamic variables using customer_requirements and notes")

        prompts = load_all_prompts()

# Build final LLM prompt (notes preferred, greeting fallback)
        if notes:
            llm_prompt = f"{prompts}\n\nCustomer Notes:\n{notes}"
        else:
            llm_prompt = f"{prompts}\n\nGreeting:\nHello, thank you for taking the time to speak with us."


        details = f"These are the details of the customer you are speaking with. Name: {customer_name}:\n\n"
        details += f"Customer Requirements: {customer_requirements}\n"
        details += f"Notes: {notes}\n"
        details += f"Tasks: {tasks}\n"
        details += f"Company Name: {company_name}\n"
        details += f"Country Code: {country_code}\n"
        details += f"Industry: {industry}\n"
        details += f"Location: {location}\n"
        details += f"LLM Prompt:\n{llm_prompt}\n"

        logger.info(f"[process_queue_single_run] Details for call: {details}\n\n")
        # Normalize phone number
        phone = phone_number.strip()
        logger.info(f"[process_queue_single_run] Trying Phone: {phone}\n\n")
        
        # If no valid phone found, remove from queue and restart thread
        if not phone:
            logger.error(f"[process_queue_single_run] No valid phone found for {customer_id} (call_id: {call_id}). Removing from queue.\n\n")
            pop_call_by_id(call_id)
            threading.Thread(target=process_queue_single_run, daemon=True).start()
            return


        correlation_id = str(uuid.uuid4())
        try:
            call_success = initiate_call(
                phone_number=phone,
                details=details,
                lead_name=customer_name,
                customer_id=customer_id,
                correlation_id=correlation_id,
                call_id=call_id,
                email=email,
                country_code=country_code
            )
        except Exception as call_exc:
            logger.error(f"[process_queue_single_run] Exception during call initiation for call_id {call_id}: {call_exc}\n\n", exc_info=True)
            call_success = False

        if not call_success:
            logger.error(f"[process_queue_single_run] Call initiation failed for {customer_id} (call_id: {call_id}). Removing from queue.\n\n")
            pop_call_by_id(call_id)
            threading.Thread(target=process_queue_single_run, daemon=True).start()
        else:
            logger.info(f"[process_queue_single_run] Call successfully initiated for {customer_id} (call_id: {call_id}). Awaiting webhook or Twilio polling.\n\n")

    except Exception as e:
        logger.error(f"[process_queue_single_run] Unexpected error: {e}\n\n", exc_info=True)
        if 'call_id' in locals():
            pop_call_by_id(call_id)


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

    # Append new prompt (don't overwrite existing)
    with open(PROMPT_FILE, "a", encoding="utf-8") as f:
        f.write(prompt_text.strip() + "\n")

    return {"status": "success", "message": "Prompt saved successfully"}

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...), username: str = Depends(get_current_user)):
    logger.info(f"[upload_file API] User {username} uploading file: {file.filename}\n\n")
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
        raise HTTPException(status_code=500, detail="Failed to generate Excel file.")
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
    logger.info(f"[add_call API] User {username} processing previously uploaded Excel file\n\n")
    TEMP_FILE_PATH = "temp_upload.xlsx"
    try:
        if not os.path.exists(TEMP_FILE_PATH):
            logger.error("No file uploaded yet. temp_upload.xlsx not found.")
            raise HTTPException(status_code=400, detail="No file uploaded yet. Please upload an Excel file first.")

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
            customer_name_raw = row.get('customer_name', '')
            if pd.isna(customer_name_raw):
                customer_name = None
            else:
                customer_name = str(customer_name_raw).strip()
            customer_id = (str(row.get('customer_id', '')) if row.get('customer_id', '') is not None else '').strip()
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
            email = (str(row.get('email', '')) if row.get('email', '') is not None else '').strip()
            customer_requirements = (str(row.get('customer_requirements', '')) if row.get('customer_requirements', '') is not None else '').strip()
            notes = (str(row.get('notes', '')) if row.get('notes', '') is not None else '').strip()
            tasks = (str(row.get('tasks', '')) if row.get('tasks', '') is not None else '').strip()
            to_call = (str(row.get('to_call', '')) if row.get('to_call', '') is not None else '').strip()
            industry = (str(row.get('industry', '')) if row.get('industry', '') is not None else '').strip()
            company_name = (str(row.get('company_name', '')) if row.get('company_name', '') is not None else '').strip()
            location = (str(row.get('location', '')) if row.get('location', '') is not None else '').strip()
            logger.info(f"Processing row: customer_id={customer_id}, customer_name={customer_name}, phone_number={phone_number}, to_call={to_call}")
            if to_call.lower() == "yes":
                # Insert into call_queue first to get call_id
                if customer_name and phone_number:
                    try:
                        c.execute("INSERT INTO call_queue (customer_name, customer_id, phone_number, email, customer_requirements, to_call, notes, tasks, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued')", (customer_name, customer_id, phone_number, email, customer_requirements, to_call, notes, tasks))
                        call_id = c.lastrowid
                        # Insert into customer_data with the same call_id
                        c.execute("""
                            INSERT OR REPLACE INTO customer_data (call_id, customer_name, customer_id, phone_number, email, customer_requirements, to_call, notes, tasks, country_code, industry, company_name, location)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (call_id, customer_name, customer_id, phone_number, email, customer_requirements, to_call, notes, tasks, country_code, industry, company_name, location))
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
        raise HTTPException(status_code=400, detail="Failed to process Excel file.")



@app.post("/webhook/call-ended")
async def call_ended(request: Request):
    logger.info("[call_ended API] Received call end webhook.\n\n")

    try:
        data = await request.json()
        logger.info(f"[call_ended] Webhook received data.")
        
        # Log all top-level keys and important subfields for debugging
        logger.info(f"[call_ended] Webhook received keys: {list(data.keys())}")
        logger.info(f"[call_ended] data['data'] keys: {list(data.get('data', {}).keys())}")
        logger.info(f"[call_ended] dynamic_variables: {list(data.get('data', {}).get('conversation_initiation_client_data', {}).get('dynamic_variables', {}).keys())}")
        logger.info(f"[call_ended] analysis keys: {list(data.get('data', {}).get('analysis', {}).keys())}")
        logger.info(f"[call_ended] metadata keys: {list(data.get('data', {}).get('metadata', {}).keys())}")

        # Extract relevant data
        dynamic_vars = data["data"]["conversation_initiation_client_data"]["dynamic_variables"]
        call_id = dynamic_vars.get("call_id")
        analysis = data["data"].get("analysis", {})

        customer_id = dynamic_vars.get("customer_id")
        customer_name = dynamic_vars.get("customer_name", "Unknown Customer")
        call_summary = analysis.get("transcript_summary", "No summary provided")
        call_sid = data["data"]["metadata"]["phone_call"]["call_sid"]
        call_transcript = data["data"].get("transcript")

        logger.info(f"[call_ended] Extracted fields: call_sid={call_sid}, customer_id={customer_id}, customer_name={customer_name}, call_summary={'present' if call_summary else 'missing'}, call_transcript={'present' if call_transcript else 'missing'}")

        customer_email = dynamic_vars.get("email", "No email provided")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE customer_data SET last_call_status = ? WHERE call_id = ?", ("completed", call_id))
        conn.commit()
        conn.close()

        # Save transcript to global email_and_transcript dict
        with email_and_transcript_lock:
            if str(call_sid) not in email_and_transcript:
                email_and_transcript[str(call_sid)] = {"email": customer_email, "transcript": call_transcript}
            else:
                email_and_transcript[str(call_sid)]["transcript"] = call_transcript

        if not customer_id:
            logger.error("Missing 'customer_id' in webhook payload.\n\n")
            raise HTTPException(status_code=400, detail="Missing customer_id in webhook.")

        # Post call summary to Excel
        parsed = summarize_conversation_transcript(call_transcript)
        update_customer_data_notes_and_tasks(call_id=call_id, parsed=parsed, db_path=DB_PATH)
        send_meeting_invite(parsed=parsed, customer_name=customer_name, customer_email=customer_email)
        # Remove completed call from queue
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT call_id FROM call_queue WHERE customer_id = ? AND status = 'processing'",
                    (customer_id,)
                )
                row = cursor.fetchone()
                if row:
                    queue_id = row[0] if isinstance(row, (tuple, list)) else row
                    pop_call_by_id(queue_id)
                    logger.info(f"Removed queue entry {queue_id} for customer_id {customer_id}.\n\n")
                else:
                    logger.warning(f"No processing entry found for customer_id: {customer_id}. Possibly already handled.\n\n")
        except Exception as db_exc:
            logger.error(f"Error during queue cleanup for {customer_id}: {db_exc}\n\n", exc_info=True)

        # Trigger next call
        logger.info("[call_ended] Triggering next call after webhook.\n\n")
        threading.Thread(target=process_queue_single_run, daemon=True).start()

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
                    logger.warning(f"Stuck call detected (ID {stuck_call_id}), removing from queue.\n\n")

                    pop_call_by_id(stuck_call_id)

                    parsed = summarize_conversation_transcript(call_transcript)
                    update_customer_data_notes_and_tasks(call_id=stuck_call_id, parsed=parsed, db_path=DB_PATH)
                    send_meeting_invite(parsed=parsed, customer_name=customer_name, customer_email=customer_email)
                   
                    threading.Thread(target=process_queue_single_run, daemon=True).start()
        except Exception as stuck_exc:
            logger.error(f"Error checking for stuck calls: {stuck_exc}\n\n", exc_info=True)

        return {"status": "Webhook processed, queue updated.", "entity_id_processed": customer_id}

    except Exception as e:
        logger.error(f"Fatal error in call-ended webhook: {e}\n\n", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/status")
def queue_status():
    logger.info("[status API] /status endpoint called. Returns current queue status.\n\n")
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
        raise HTTPException(status_code=500, detail="Failed to fetch queue status.")

@app.get("/customer-data-status")
def customer_data_status():
    logger.info("[status API] /customer-data-status endpoint called. Returns current customer data status.\n\n")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("""
                SELECT call_id, customer_id, customer_name, phone_number, email, country_code
                FROM customer_data
                ORDER BY call_id ASC
            """)
            queue = [
                {
                    "call_id": row[0],
                    "customer_id": row[1],
                    "customer_name": row[2],
                    "phone": row[3],
                    "email": row[4],
                    "country_code": row[5]
                }
                for row in cursor.fetchall()
            ]
        return {"queue": queue}
    except Exception as e:
        logger.error("Error in /customer-data-status: %s\n\n", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch queue status.")

@app.post("/update-queue")
def update_queue(req: QueueUpdateRequest):
    logger.info(f"[update_queue API] /update-queue endpoint called. Updates queue entry with id: {req.id}.\n\n")
    
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
    logger.info(f"[delete_queue_item API] /delete-queue/{queue_id} endpoint called. Deleting queue item {queue_id}.\n\n")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM call_queue WHERE call_id = ?", (queue_id,))
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Queue item not found")
            conn.commit()

        # Trigger background process to handle next call
        threading.Thread(target=process_queue_single_run, daemon=True).start()

        return {"message": f"Queue item {queue_id} deleted successfully."}

    except HTTPException:
        raise  # re-raise for FastAPI to handle properly
    except Exception as e:
        logger.error(f"Error in /delete-queue/{queue_id}: {e}\n\n", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete queue item.")
        
@app.get("/delete-all-queue")
def delete_all_queue():
    logger.info("[delete_all_queue API] /delete-all-queue endpoint called. Deleting all queue items.\n\n")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM call_queue")
            conn.commit()

        return {"message": "All queue items deleted successfully."}

    except Exception as e:
        logger.error("Error in /delete-all-queue: %s\n\n", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete all queue items.")
    
@app.get("/delete-customer-data-queue")
def delete_customer_data_queue():
    logger.info("[delete_customer_data_queue API] /delete-customer-data-queue endpoint called. Deleting all customer data.\n\n")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM customer_data")
            conn.commit()

        return {"message": "All customer data deleted successfully."}

    except Exception as e:
        logger.error("Error in /delete-customer-data-queue: %s\n\n", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete all customer data.")

def cleanup_stuck_calls():
    logger.info("[cleanup_stuck_calls] Background thread started. Periodically cleaning up stuck calls.\n\n")

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
                    logger.warning(f"[{call_id}] No response for 15 minutes. Logging and retrying next call.\n\n")

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
                    #         lead_name = data.get("Name", lead_name)
                    # except Exception as fetch_exc:
                    #     logger.error(f"[{call_id}] Failed to fetch entity details: {fetch_exc}\n\n")


                    # Retrieve call_sid for this stuck call if possible
                    stuck_call_sid = None
                    with call_id_to_sid_lock:
                        stuck_call_sid = call_id_to_sid.get(str(call_id))                    
                    # Get transcript and email from global dict if possible
                    with email_and_transcript_lock:
                        stuck_email = None
                        stuck_transcript = None
                        if stuck_call_sid and stuck_call_sid in email_and_transcript:
                            stuck_email = email_and_transcript[stuck_call_sid].get("email", "No email provided")
                            stuck_transcript = email_and_transcript[stuck_call_sid].get("transcript", None)
                        else:
                            stuck_email = "No email provided"
                            stuck_transcript = None

                    parsed = summarize_conversation_transcript(stuck_transcript)
                    update_customer_data_notes_and_tasks(call_id=call_id, parsed=parsed, db_path=DB_PATH)
                    send_meeting_invite(parsed=parsed, customer_name=customer_name, customer_email=stuck_email)

                    # Start next call
                    threading.Thread(target=process_queue_single_run, daemon=True).start()

        except Exception as e:
            logger.error(f"Error in stuck call cleanup loop: {e}\n\n", exc_info=True)

        time.sleep(60)  # Wait before next check

# def periodic_queue_processor():
#     logger.info("[periodic_queue_processor] Background thread started. Periodically processing queue.\n\n")
    
#     while True:
#         try:
#             process_queue_single_run()
#         except Exception as e:
#             logger.error(f"Error in periodic queue processor: {e}\n\n", exc_info=True)

#         time.sleep(60)

# Start both background threads at app startup
# threading.Thread(target=cleanup_stuck_calls, daemon=True, name="StuckCallCleaner").start()
# threading.Thread(target=periodic_queue_processor, daemon=True, name="QueueProcessor").start()

# Periodic thread to poll /excel-status and log notification when Excel file is ready

# def poll_excel_status(interval=10):
#     """
#     Periodically polls /excel-status endpoint and logs notification when Excel file is available.
#     """
#     notified = False
#     url = "http://localhost:8001/excel-status"
#     while True:
#         try:
#             response = requests.get(url)
#             if response.status_code == 200 and response.headers.get("content-type", "").startswith("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"):
#                 if not notified:
#                     logger.info("[poll_excel_status] Resultant Excel file is now available!")
#                     notified = True
#             else:
#                 logger.info("[poll_excel_status] Excel file not ready yet.")
#                 notified = False
#         except Exception as e:
#             logger.error(f"[poll_excel_status] Error polling excel status: {e}")
#         time.sleep(interval)

# # Start the polling thread at app startup
# threading.Thread(target=poll_excel_status, daemon=True, name="ExcelStatusPoller").start()