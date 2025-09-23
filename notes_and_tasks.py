import os
from dotenv import load_dotenv
from groq import Groq
import json
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
import dateparser
import sqlite3
import pandas as pd
from datetime import datetime
from logger_config import logger


load_dotenv()

# === Configuration ===
Groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=Groq_api_key)


def summarize_conversation_transcript(conversation_transcript):
    # Handle case where transcript is a list of dicts (preserve roles/messages)
    if isinstance(conversation_transcript, list):
        formatted_transcript = []
        for turn in conversation_transcript:
            role = turn.get("role", "")
            message = turn.get("message", "")
            formatted_transcript.append(f"{role}: {message}")
        conversation_transcript = "\n".join(formatted_transcript)
    print("Formatted Transcript:", conversation_transcript)  # Debugging line

    if not conversation_transcript or not isinstance(conversation_transcript, str):
        logger.error(
            "[summarize_conversation_transcript] Conversation transcript is empty or None.")
        return {
            "summary": "",
            "tasks": "",
            "meeting_schedule_is_true": False,
            "meeting_type_in_person": False,
            "meeting_type_virtual": False,
            "meeting_time_in_person_raw": "",
            "meeting_time_virtual_raw": ""
        }
    # --- Chunking logic ---

    def chunk_text(text, chunk_size=1000, overlap=100):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i+chunk_size]
            chunks.append(' '.join(chunk))
            i += chunk_size - overlap
        return chunks

    chunks = chunk_text(conversation_transcript, chunk_size=1000, overlap=100)
    logger.info(
        f"[summarize_conversation_transcript] Split transcript into {len(chunks)} chunks.\n\n")

    all_summaries = []
    all_tasks = []
    meeting_schedule_is_true = False
    meeting_type_in_person = False
    meeting_type_virtual = False
    meeting_time_in_person_raw = ""
    meeting_time_virtual_raw = ""

    for idx, chunk in enumerate(chunks):
        system_prompt = """You are a helpful assistant that extracts summary and tasks from AI call transcripts. 
        The conversation transcript is provided by the user.
        The details of the tasks should be detailed and descriptive. 
        Make sure to include all tasks mentioned in the conversation chunk.
        Make sure you **do not** miss any tasks and/or their details.
        **Do not** take normal conversation where the customer is expressing their requirements as a task. Only take the task if the customer explicitly wants the company to perform an action.
        from the transcript, you have to summarize the conversation in short.
        from the transcript, you have to extract the tasks that the customer asks to perform.
        Only extract tasks that the customer/user explicitly wants the company to perform. Ignore greetings, agent actions, and generic conversational flow.
        from the transcript, you have to check if the customer wants to schedule a meeting with the owner.
        if the customer wants to schedule a meeting, you have to return the meeting status as True.
        if the customer wants to schedule an in-person meeting, you have to return meeting_type_in_person in JSON response with a boolean value True. If the customer does not want to schedule an in-person meeting, you have to return meeting_type_in_person as False.
        "In-person meeting" means the customer wants to visit a showroom or a store or wants to have a meeting at a physical location.
        if the customer wants to schedule an in person meeting, then find out the time that the user prefers for the in-person meeting and return it in the JSON response as meeting_time_in_person_raw.
        if the customer wants to schedule a virtual meeting, you have to return meeting_type_virtual in JSON response with a boolean value True. If the customer does not want to schedule a virtual meeting, you have to return meeting_type_virtual as False.
        if the customer wants to schedule a virtual meeting, then find out the time that the user prefers for the virtual meeting and return it in the JSON response as meeting_time_virtual_raw.
        if the customer wants to schedule both a virtual and an in-person meeting, you have to return both meeting_type_in_person and meeting_type_virtual as True. In this case, you have to find out the time that the user prefers for both meetings and return it in the JSON response as meeting_time_in_person_raw and meeting_time_virtual_raw.
        if the customer does not want to schedule a meeting, you have to return the meeting status as False and also return meeting_type_in_person and meeting_type_virtual as False. In this case, you do not need to return meeting_time_in_person_raw and meeting_time_virtual_raw. Just return them as empty strings. But it is applicable only if the customer does not want to schedule a meeting.
        if meeting_type_in_person is True, then meeting_time_in_person_raw should not be empty.
        if meeting_type_virtual is True, then meeting_time_virtual_raw should not be empty.
        If meeting_type_in_person is False, then meeting_time_in_person_raw should be an empty string.
        If meeting_type_virtual is False, then meeting_time_virtual_raw should be an empty string.
        If both meeting_type_in_person and meeting_type_virtual are False, then both meeting_time_in_person_raw and meeting_time_virtual_raw should be empty strings.
        If both meeting_type_in_person and meeting_type_virtual are True, then both meeting_time_in_person_raw and meeting_time_virtual_raw should contain the respective preferred times.
        your final JSON response should look like this (this is just an example, do not use these values.):\n\n
        {
            "summary": "the detailed summary of the conversation",
            "tasks": 1. Get the product details , 2. Call Tom , ...,
            "meeting_schedule_is_true": true/false,
            "meeting_type_in_person": true/false,
            "meeting_type_virtual": true/false,
            "meeting_time_in_person_raw": "in person raw meeting time as string",
            "meeting_time_virtual_raw": "virtual meeting time as string"
        }
        Make sure to return the JSON response in a single line without any extra spaces or newlines.
        Do not return any other text or explanation. Just return the JSON response as it is.
        Do not return ````json`` or any other formatting. Just return the JSON response as it is."""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""Here is the conversation transcript chunk {idx+1}/{len(chunks)}:\n\n
                    f"{chunk}\n\n
                    "Please extract any summary,tasks, meeting_schedule_is_true, meeting_time and meeting_type mentioned by the customer in this conversation transcript chunk.
                    "Only return valid JSON.
                                """
            }
        ]

        try:
            logger.info(
                f"[summarize_conversation_transcript] Summarizing chunk {idx+1}/{len(chunks)}.\n\n")
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
            )
            logger.info(
                "[summarize_conversation_transcript] Received response from LLM.\n\n")
            content = response.choices[0].message.content.strip()
            logger.info(
                f"[summarize_conversation_transcript] Raw content from LLM: {content}\n\n")
            if not content.startswith('{') or not content.endswith('}'):
                logger.error(
                    "[summarize_conversation_transcript] Response content is not a valid JSON object.\n\n")
                continue
            content = content.replace("\n", "").replace("\t", "")
            content = content.replace("True", "true").replace("False", "false")
            parsed = json.loads(content)
            logger.info(
                f"[summarize_conversation_transcript] Parsed JSON (chunk {idx+1}): {parsed}\n\n")

            # Collect summaries and tasks
            if "summary" in parsed:
                all_summaries.append(parsed["summary"])
            if "tasks" in parsed:
                all_tasks.append(str(parsed["tasks"]))

            # Merge meeting info if found in any chunk
            if parsed.get("meeting_schedule_is_true", False):
                meeting_schedule_is_true = True
            if parsed.get("meeting_type_in_person", False):
                meeting_type_in_person = True
                meeting_time_in_person_raw = parsed.get(
                    "meeting_time_in_person_raw", "")
            if parsed.get("meeting_type_virtual", False):
                meeting_type_virtual = True
                meeting_time_virtual_raw = parsed.get(
                    "meeting_time_virtual_raw", "")

        except json.JSONDecodeError as e:
            logger.error(
                f"[summarize_conversation_transcript] JSON decoding failed for chunk {idx+1}: {e}\n\n")
            continue
        except Exception as e:
            logger.error(
                f"[summarize_conversation_transcript] Unexpected error in chunk {idx+1}: {e}\n\n")
            continue

    # Combine all summaries
    combined_summary = "\n".join(all_summaries)

    # Process and combine all tasks into a single numbered list
    task_list = []
    for chunk_tasks in all_tasks:
        # Split by line or numbered pattern
        lines = [t.strip() for t in chunk_tasks.split('\n') if t.strip()]
        for line in lines:
            # Remove leading numbering (e.g., '1. ', '2. ', etc.)
            cleaned = line.lstrip('0123456789. ').strip()
            if cleaned and cleaned.lower() not in [t.lower() for t in task_list]:
                task_list.append(cleaned)

    # Format as a single numbered list
    combined_tasks = "\n".join(
        [f"{i+1}. {task}" for i, task in enumerate(task_list)])

    final_parsed_result = {
        "summary": combined_summary,
        "tasks": combined_tasks,
        "meeting_schedule_is_true": meeting_schedule_is_true,
        "meeting_type_in_person": meeting_type_in_person,
        "meeting_type_virtual": meeting_type_virtual,
        "meeting_time_in_person_raw": meeting_time_in_person_raw,
        "meeting_time_virtual_raw": meeting_time_virtual_raw
    }
    logger.info(
        f"[summarize_conversation_transcript] Final combined output: {final_parsed_result}\n\n")
    return final_parsed_result


def update_customer_data_notes_and_tasks(call_id, parsed, db_path="queue.db"):
    """
    Appends new notes and tasks (with timestamp) to the existing notes and tasks columns for a given call_id in customer_data.
    """
    logger.info(f"[update_customer_data_notes_and_tasks] Starting update for call_id: {call_id}")
    logger.info(f"[update_customer_data_notes_and_tasks] Parsed data: {parsed}")

    if not parsed:
        logger.warning(
            f"[update_customer_data_notes_and_tasks] No valid parsed data available for call_id: {call_id}\n\n")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("SELECT notes, tasks FROM customer_data WHERE call_id = ?", (call_id,))
    row = c.fetchone()
    if row:
        existing_notes, existing_tasks = row
        logger.info(f"[update_customer_data_notes_and_tasks] Existing notes: {existing_notes}")
        logger.info(f"[update_customer_data_notes_and_tasks] Existing tasks: {existing_tasks}")
        
        new_notes = "No summary available. Conversation transcript missing." if not parsed else parsed.get(
            "summary", "No summary available.")
        new_tasks = "No tasks found for this call." if not parsed else parsed.get(
            "tasks", "No tasks found for this call.")

        # Clean up "nan" strings and empty values
        def clean_value(value):
            if value is None:
                return ""
            value_str = str(value).strip().lower()
            if value_str in ["nan", "null", "none", ""]:
                return ""
            return value

        new_notes = clean_value(new_notes)
        new_tasks = clean_value(new_tasks)

        # Only add timestamp if we have actual content
        notes_prefix = f"\n[{timestamp}] " if new_notes else ""
        tasks_prefix = f"\n[{timestamp}] " if new_tasks else ""

        updated_notes = (existing_notes or "") + notes_prefix + new_notes
        updated_tasks = (existing_tasks or "") + tasks_prefix + new_tasks

        c.execute("UPDATE customer_data SET notes = ?, tasks = ? WHERE call_id = ?",
                  (updated_notes, updated_tasks, call_id))
        conn.commit()
    conn.close()


def export_customer_data_to_excel(db_path="queue.db", excel_path="resultant_excel.xlsx"):
    """
    Exports the entire customer_data table to resultant_excel.xlsx.
    Use this after all calls are processed to get the final Excel.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM customer_data", conn)
    df.to_excel(excel_path, index=False)
    conn.commit()
    conn.close()


def send_meeting_invite(parsed, customer_name, customer_email):
    """
    Sends a meeting invite based on the parsed data.
    """
    if not parsed:
        logger.error("[send_meeting_invite] No parsed data provided")
        return

    meeting_status = parsed.get("meeting_schedule_is_true", False)
    logger.info(f"[send_meeting_invite] Meeting status: {meeting_status}")
    meeting_status_in_person = parsed.get("meeting_type_in_person", False)
    logger.info(
        f"[send_meeting_invite] In-person meeting status: {meeting_status_in_person}")
    meeting_status_virtual = parsed.get("meeting_type_virtual", False)
    logger.info(
        f"[send_meeting_invite] Virtual meeting status: {meeting_status_virtual}")
    meeting_time_in_person_raw = parsed.get("meeting_time_in_person_raw", "")
    meeting_time_virtual_raw = parsed.get("meeting_time_virtual_raw", "")

    if not meeting_status:
        logger.info("[send_meeting_invite] No meeting scheduled")
        return

    logger.info(
        f"[send_meeting_invite] Meeting types - In-person: {meeting_status_in_person}, Virtual: {meeting_status_virtual}")

    # Handle in-person meeting invite
    if meeting_status_in_person:
        try:
            current_datetime_est = datetime.now(ZoneInfo("America/New_York"))
            meeting_time_in_person_est = dateparser.parse(
                meeting_time_in_person_raw,
                settings={
                    'RELATIVE_BASE': current_datetime_est,
                    'TIMEZONE': 'America/New_York',
                    'RETURN_AS_TIMEZONE_AWARE': True
                }
            )

            logger.info(
                f"[send_meeting_invite] Parsed in-person meeting time: {meeting_time_in_person_est}")
            if meeting_time_in_person_est is not None:
                meeting_time_in_person = meeting_time_in_person_est.isoformat()
                response = requests.post(
                    headers={"Content-Type": "application/json"},
                    url="https://func-send-calendar-invite-tm-dev-fhdcbce9ebdpcmcg.eastus-01.azurewebsites.net/api/schedule_meeting",
                    json={
                        "attendee_email": customer_email,
                        "attendee_name": customer_name,
                        "subject": "In-person Meeting invite from Technology Mindz",
                        "body": """Hi, we're really grateful that you could give us some of your precious time.
                                We'll make sure it's worth your while.""",
                        "start_time": meeting_time_in_person,
                        "duration_minutes": 30,
                        "meeting_type": "in_person"
                    }
                )
                response.raise_for_status()
                logger.info(
                    "[send_meeting_invite] In-person calendar invite sent successfully")
            else:
                logger.error(
                    f"[send_meeting_invite] Could not parse in-person meeting time: '{meeting_time_in_person_raw}'")
        except Exception as e:
            logger.error(
                f"[send_meeting_invite] Error sending in-person meeting invite: {str(e)}")

    # Handle virtual meeting invite
    if meeting_status_virtual:
        try:
            current_datetime_est = datetime.now(ZoneInfo("America/New_York"))
            meeting_time_virtual_est = dateparser.parse(
                meeting_time_virtual_raw,
                settings={
                    'RELATIVE_BASE': current_datetime_est,
                    'TIMEZONE': 'America/New_York',
                    'RETURN_AS_TIMEZONE_AWARE': True
                }
            )

            logger.info(
                f"[send_meeting_invite] Parsed virtual meeting time: {meeting_time_virtual_est}")
            if meeting_time_virtual_est is not None:
                meeting_time_virtual = meeting_time_virtual_est.isoformat()
                response = requests.post(
                    headers={"Content-Type": "application/json"},
                    url="https://func-send-calendar-invite-tm-dev-fhdcbce9ebdpcmcg.eastus-01.azurewebsites.net/api/schedule_meeting",
                    json={
                        "attendee_email": customer_email,
                        "attendee_name": customer_name,
                        "subject": "Virtual Meeting invite from Technology Mindz",
                        "body": """Hi, we're really grateful that you could give us some of your precious time.
                                We'll make sure it's worth your while.""",
                        "start_time": meeting_time_virtual,
                        "duration_minutes": 30,
                        "meeting_type": "virtual"
                    }
                )
                response.raise_for_status()
                logger.info(
                    "[send_meeting_invite] Virtual calendar invite sent successfully")
            else:
                logger.error(
                    f"[send_meeting_invite] Could not parse virtual meeting time: '{meeting_time_virtual_raw}'")
        except Exception as e:
            logger.error(
                f"[send_meeting_invite] Error sending virtual meeting invite: {str(e)}")

    logger.info("[send_meeting_invite] Calendar invite processing completed")
