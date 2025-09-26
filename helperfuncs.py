import sqlite3
from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from logger_config import logger
from config import settings
from groq import Groq



# Database path
DB_PATH = "queue.db"

# --- DB Setup ---
def init_db(logger):
    logger.info("[init_db] Initializing the call_queue and customer_data databases and ensuring schema.")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Call queue for process management
    c.execute('''
        CREATE TABLE IF NOT EXISTS call_queue (
            call_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT,
            customer_id TEXT,
            phone_number TEXT,
            email TEXT,
            customer_requirements TEXT,
            specific_prompt TEXT,
            to_call TEXT,
            notes TEXT,
            tasks TEXT,
            status TEXT DEFAULT 'queued',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            called_at TIMESTAMP
        )
    ''')
    # Persistent customer data for notes/tasks/results
    c.execute('''
            CREATE TABLE IF NOT EXISTS customer_data (
                call_id INTEGER PRIMARY KEY,
                customer_id TEXT,
                customer_name TEXT,
                phone_number TEXT UNIQUE,
                email TEXT,
                customer_requirements TEXT,
                specific_prompt TEXT,
                last_call_status TEXT,
                country_code TEXT,
                industry TEXT,
                company_name TEXT,
                location TEXT,
                to_call TEXT,
                notes TEXT,
                tasks TEXT
            )
        ''')
    conn.commit()
    conn.close()

# --- Models ---
class CallRequest(BaseModel):

    type: str = Field(..., description="lead or opportunity or contact", example="lead")
    ids: list[str] = Field(..., description="List of Entity IDs", example=["001xx000003DGb1AAG", "001xx000003DGb2AAG"])
    # start_times: Optional[list[str]] = Field(None, 
    #                                   description="List of start_time strings aligned with ids (24-hour, e.g. '09:00' or '09:00:00')",
    #                                   example=["09:00", "10:00"])
    # end_times: Optional[list[str]] = Field(None,
    #                                 description="List of end_time strings aligned with ids (24-hour, e.g. '17:00' or '17:00:00')",
    #                                 example=["17:00", "18:00"])
    # re_engage_values: Optional[list[str]] = Field(None,
    #                                 description="Re-engagement value to set for the call queue entries",
    #                                 example=["re-engage after 3 days"])
    start_times: Optional[str] = Field(None,
                                      description="Start time for the call window (24-hour format, e.g. '09:00' or '09:00:00')",
                                        example="09:00")
    end_times: Optional[str] = Field(None,
                                      description="End time for the call window (24-hour format, e.g. '17:00' or '17:00:00')",
                                        example="17:00")
    re_engage_values: Optional[str] = Field(None,
                                      description="Re-engagement value to set for the call queue entries",
                                        example="re-engage after 3 days")
    specific_prompt: Optional[str] = Field(None, description="Customer-specific prompt")  # NEW FIELD
class QueueUpdateRequest(BaseModel):
    id: int = Field(..., description="Queue entry ID")
    status: Optional[str] = Field(None, description="New status for the call (queued, processing, called)")
    phone_number: Optional[str] = Field(None, description="Update phone number")
    lead_name: Optional[str] = Field(None, description="Update lead name")
    details: Optional[str] = Field(None, description="Update details")

# --- Core DB Functions ---
def add_to_queue(entity_type: str, entity_id: str) -> bool:
    logger.info(f"[add_to_queue] Attempting to add {entity_type}:{entity_id} to the queue.")
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO call_queue (entity_type, entity_id) VALUES (?, ?)",
                  (entity_type, entity_id))
        conn.commit()
        logger.info(f"Added {entity_type}:{entity_id} to queue.")
        return True
    except sqlite3.IntegrityError:
        logger.warning(f"[{entity_type}:{entity_id}] already exists in queue. Skipping addition.")
        return False
    except Exception as e:
        logger.error(f"Error adding to queue: {e}")
        raise HTTPException(status_code=500, detail="Failed to add to queue")
    finally:
        conn.close()

def pop_next_call():
    logger.info("[pop_next_call] Attempting to fetch and mark the next queued call as processing.")
    conn = sqlite3.connect(DB_PATH, isolation_level='EXCLUSIVE')  # lock DB during transaction
    try:
        c = conn.cursor()

        # Start transaction
        c.execute("BEGIN EXCLUSIVE")
        c.execute("""
            SELECT call_id, customer_name, customer_id, phone_number, email, customer_requirements, notes, tasks, specific_prompt
            FROM call_queue
            WHERE status = 'queued'
            ORDER BY created_at ASC
            LIMIT 1
        """)
        row = c.fetchone()

        if row:
            call_id, customer_name, customer_id, phone_number, email, customer_requirements, notes, tasks, specific_prompt = row
            # Mark as processing
            c.execute("""
                UPDATE call_queue
                SET status = 'processing', called_at = CURRENT_TIMESTAMP
                WHERE call_id = ?
            """, (call_id,))
            conn.commit()
            logger.info(f"Marked call_id {call_id} as processing.")
            return call_id, customer_name, customer_id, phone_number, email, customer_requirements, notes, tasks, specific_prompt
        else:
            logger.info("No queued calls found.")
            conn.commit()
            return None

    except sqlite3.Error as e:
        logger.error(f"Error in pop_next_call: {e}", exc_info=True)
        conn.rollback()
        return None
    finally:
        conn.close()


def update_call_details(call_id: int, phone_number: str, lead_name: str, details: str):
    logger.info(f"[update_call_details] Updating call details for call_id: {call_id}.")
    """Updates the phone number, lead name, and details for a specific call queue entry."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE call_queue SET phone_number = ?, lead_name = ?, details = ? WHERE call_id = ?",
                  (phone_number, lead_name, details, call_id))
        conn.commit()
        logger.info(f"Updated details for call_id: {call_id}")
    except Exception as e:
        logger.error(f"Error updating call details for call_id {call_id}: {e}")
        raise RuntimeError(f"Failed to update call details: {e}")
    finally:
        conn.close()

def mark_call_completed(call_id: int):
    logger.info(f"[mark_call_completed] Marking call_id {call_id} as 'called'.")
    """Marks a call queue entry as 'called'."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE call_queue SET status = 'called' WHERE id = ?", (call_id,))
        conn.commit()
        logger.info(f"Marked call_id {call_id} as 'called'.")
    except Exception as e:
        logger.error(f"Error marking call_id {call_id} completed: {e}")
        raise RuntimeError(f"Failed to mark call completed: {e}")
    finally:
        conn.close()

def pop_call_by_id(call_id: int):
    logger.info(f"[pop_call_by_id] Removing call_id {call_id} from the queue.")
    """Deletes a call queue entry by its CALL_ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM call_queue WHERE call_id = ?", (call_id,))
        conn.commit()
        if c.rowcount > 0:
            logger.info(f"Removed call_id {call_id} from queue.")
        else:
            logger.warning(f"Attempted to remove call_id {call_id}, but it was not found.")
    except Exception as e:
        logger.error(f"Error popping call_id {call_id}: {e}")
    finally:
        conn.close()

# def fetch_contact_details(contact_id: str):
#     logger.info(f"[fetch_contact_details] Fetching Salesforce contact details for contact_id: {contact_id}.")
#     formatter = SalesforceContactFormatter()
#     try:
#         contact_info = formatter.get_formatted_contact_info(contact_id)
#         return contact_info
#     except Exception as e:
#         logger.error(f"Error fetching contact details: {e}")
#         raise HTTPException(status_code=500, detail="Failed to fetch contact details")
    
COUNTRY_CODE_MAP = {
    # North America (23 entries + aliases)
    "united states": "US", "usa": "US", "us": "US", "america": "US", "us of a": "US", "united states of america": "US", "canada": "CA", "can": "CA",
    "mexico": "MX", "mex": "MX", "cuba": "CU", "dominican republic": "DO", "dr": "DO", "haiti": "HT", "jamaica": "JM", "trinidad and tobago": "TT",
    "bahamas": "BS", "barbados": "BB", "belize": "BZ", "costa rica": "CR", "el salvador": "SV", "guatemala": "GT", "honduras": "HN", "nicaragua": "NI",
    "panama": "PA", "antigua and barbuda": "AG", "aruba": "AW", "bermuda": "BM", "bonaire, sint eustatius and saba": "BQ", "bonaire": "BQ",
    "british virgin islands": "VG", "bvi": "VG", "cayman islands": "KY", "curaçao": "CW", "dominica": "DM", "grenada": "GD", "guadeloupe": "GP",
    "martinique": "MQ", "montserrat": "MS", "puerto rico": "PR", "saint barthélemy": "BL", "saint kitts and nevis": "KN", "saint lucia": "LC",
    "saint martin (french part)": "MF", "saint pierre and miquelon": "PM", "saint vincent and the grenadines": "VC",
    "sint maarten (dutch part)": "SX", "turks and caicos islands": "TC", "united states virgin islands": "VI", "us virgin islands": "VI",

    # South America (14 entries + aliases)
    "argentina": "AR", "arg": "AR", "brazil": "BR", "bra": "BR", "colombia": "CO", "col": "CO", "chile": "CL", "peru": "PE",
    "venezuela": "VE", "ecuador": "EC", "bolivia": "BO", "paraguay": "PY", "uruguay": "UY", "guyana": "GY", "suriname": "SR",
    "french guiana": "GF", "falkland islands (malvinas)": "FK", "falkland islands": "FK", "south georgia and the south sandwich islands": "GS",

    # Europe (50 entries + aliases)
    "united kingdom": "GB", "uk": "GB", "great britain": "GB", "germany": "DE", "france": "FR", "italy": "IT", "spain": "ES", "poland": "PL",
    "ukraine": "UA", "russian federation": "RU", "russia": "RU", "netherlands": "NL", "holland": "NL", "belgium": "BE", "switzerland": "CH",
    "sweden": "SE", "norway": "NO", "denmark": "DK", "finland": "FI", "ireland": "IE", "austria": "AT", "portugal": "PT", "greece": "GR",
    "czechia": "CZ", "czech republic": "CZ", "hungary": "HU", "romania": "RO", "bulgaria": "BG", "serbia": "RS", "croatia": "HR",
    "bosnia and herzegovina": "BA", "slovenia": "SI", "slovakia": "SK", "albania": "AL", "north macedonia": "MK", "montenegro": "ME",
    "kosovo": "XK", "estonia": "EE", "latvia": "LV", "lithuania": "LT", "iceland": "IS", "luxembourg": "LU", "malta": "MT", "cyprus": "CY",
    "andorra": "AD", "monaco": "MC", "san marino": "SM", "vatican city": "VA", "holy see": "VA", "liechtenstein": "LI", "gibraltar": "GI",
    "guernsey": "GG", "jersey": "JE", "isle of man": "IM", "faroe islands": "FO", "aland islands": "AX", "svalbard and jan mayen": "SJ",

    # Asia (53 entries + aliases)
    "india": "IN", "china": "CN", "japan": "JP", "south korea": "KR", "korea": "KR", "indonesia": "ID", "pakistan": "PK", "bangladesh": "BD",
    "philippines": "PH", "vietnam": "VN", "thailand": "TH", "myanmar": "MM", "burma": "MM", "malaysia": "MY", "singapore": "SG",
    "afghanistan": "AF", "iran": "IR", "iran (islamic republic of)": "IR", "iraq": "IQ", "saudi arabia": "SA", "united arab emirates": "AE",
    "uae": "AE", "turkey": "TR", "israel": "IL", "lebanon": "LB", "jordan": "JO", "syrian arab republic": "SY", "syria": "SY", "kuwait": "KW",
    "qatar": "QA", "bahrain": "BH", "oman": "OM", "yemen": "YE", "azerbaijan": "AZ", "georgia": "GE", "armenia": "AM", "kazakhstan": "KZ",
    "uzbekistan": "UZ", "kyrgyzstan": "KG", "tajikistan": "TJ", "turkmenistan": "TM", "nepal": "NP", "sri lanka": "LK", "bhutan": "BT",
    "maldives": "MV", "brunei darussalam": "BN", "brunei": "BN", "cambodia": "KH", "lao people's democratic republic": "LA", "laos": "LA",
    "mongolia": "MN", "timor-leste": "TL", "east timor": "TL", "hong kong": "HK", "macao": "MO", "taiwan": "TW", "taiwan, province of china": "TW",
    "palestine, state of": "PS", "palestine": "PS",

    # Africa (59 entries + aliases)
    "nigeria": "NG", "egypt": "EG", "south africa": "ZA", "rsa": "ZA", "ethiopia": "ET", "kenya": "KE", "tanzania": "TZ",
    "tanzania, united republic of": "TZ", "algeria": "DZ", "sudan": "SD", "morocco": "MA", "angola": "AO", "ghana": "GH",
    "mozambique": "MZ", "madagascar": "MG", "cameroon": "CM", "côte d'ivoire": "CI", "ivory coast": "CI", "niger": "NE",
    "burkina faso": "BF", "mali": "ML", "malawi": "MW", "zambia": "ZM", "senegal": "SN", "chad": "TD", "somalia": "SO",
    "zimbabwe": "ZW", "guinea": "GN", "rwanda": "RW", "benin": "BJ", "tunisia": "TN", "burundi": "BI", "south sudan": "SS",
    "togo": "TG", "sierra leone": "SL", "libya": "LY", "congo (democratic republic of the)": "CD", "dr congo": "CD", "drc": "CD",
    "central african republic": "CF", "car": "CF", "liberia": "LR", "mauritania": "MR", "eritrea": "ER", "gambia": "GM",
    "botswana": "BW", "namibia": "NA", "gabon": "GA", "lesotho": "LS", "guinea-bissau": "GW", "equatorial guinea": "GQ",
    "mauritius": "MU", "eswatini": "SZ", "swaziland": "SZ", "djibouti": "DJ", "comoros": "KM", "cabo verde": "CV",
    "cape verde": "CV", "sao tome and principe": "ST", "seychelles": "SC", "reunion": "RE", "mayotte": "YT",
    "saint helena, ascension and tristan da cunha": "SH", "saint helena": "SH", "western sahara": "EH",

    # Oceania (26 entries + aliases)
    "australia": "AU", "aussie": "AU", "new zealand": "NZ", "nz": "NZ", "papua new guinea": "PG", "png": "PG", "fiji": "FJ",
    "solomon islands": "SB", "vanuatu": "VU", "new caledonia": "NC", "french polynesia": "PF", "samoa": "WS", "guam": "GU",
    "kiribati": "KI", "micronesia (federated states of)": "FM", "micronesia": "FM", "marshall islands": "MH", "nauru": "NR",
    "palau": "PW", "tuvalu": "TV", "tonga": "TO", "cook islands": "CK", "niue": "NU", "norfolk island": "NF",
    "northern mariana islands": "MP", "american samoa": "AS", "wallis and futuna": "WF", "tokelau": "TK",
    "christmas island": "CX", "cocos (keeling) islands": "CC", "pitcairn": "PN",

    # Antarctica (1 entry)
    "antarctica": "AQ",

    # United States Minor Outlying Islands (1 entry)
    "united states minor outlying islands": "UM", "us minor outlying islands": "UM",

    # Heard Island and McDonald Islands (1 entry)
    "heard island and mcdonald islands": "HM",

    # Bouvet Island (1 entry)
    "bouvet island": "BV",

    # British Indian Ocean Territory (1 entry)
    "british indian ocean territory": "IO",

    # French Southern Territories (1 entry)
    "french southern territories": "TF",
}

# Assuming these constants are set somewhere in your environment
DB_PATH = "queue.db"

def generate_initial_message(lead_data: str, llm_prompt: str = "") -> str:
    import time
    client = Groq(api_key=settings.GROQ_API_KEY)

    # Prepare the prompt that instructs the LLM what to do
    system_prompt = (
        "You are \"Technology Mindz's AI Assistant\" making a live phone call. Begin the call with a warm, conversational message. "
        "Use the provided lead data to personalize the greeting and briefly summarize their interest. "
        "Invite them to share more if they’re available to talk, and if not, offer to schedule a meeting instead. "
        "Avoid email-like phrasing — this should sound like natural, real-time speech suitable for a voice call."
        "Do not mention the stage at which that lead is."
        "Keep this message concise, friendly, and engaging. Form a short summary of the lead data provided, and use it to create a personalized greeting. But remember to keep the message short and engaging, as if you are speaking to the customer in real-time."
        "**IMPORTANT**"
        "Make the greeting message short and concise."
        "**Necessarily** after saying 'Hi {name of the user}, say 'This is Technology Mindz's AI assistant.' Then talk a little about what you can see in their {lead data} in not more than 10 to 20 words and then necessarily ask the user if this is the right time to talk."
       
    )
    if llm_prompt:
        enhanced_prompt = f"{system_prompt}\n\nAdditional Instructions:\n{llm_prompt}"
    else:
        enhanced_prompt = system_prompt
        
    user_prompt = f"Here is the customer data:\n{lead_data[:8000]}\nGenerate Technology Mindz's AI Assistant a first message that Technology Mindz's AI assistant would say when reaching out to understand their requirements and offer help."

    max_retries = 3
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": enhanced_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile"
            )
            # If Groq returns a valid response, break and return
            if hasattr(chat_completion, 'choices') and chat_completion.choices and hasattr(chat_completion.choices[0], 'message'):
                return chat_completion.choices[0].message.content
            else:
                logger.info(f"Groq API did not return a valid response, attempt {attempt+1}.")
        except Exception as e:
            # Check for Groq limit exceeded or non-200 response
            logger.info(f"Groq API call failed (attempt {attempt+1}): {e}. Retrying in 60 seconds...")
            time.sleep(60)
    logger.info("Failed to get a valid response from Groq after 3 attempts.")
    return "[Error: Unable to generate initial message due to Groq API limit or error.]"
