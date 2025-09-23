from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ELEVENLABS_API: str
    ELEVENLABS_WEBHOOK_SECRET: str
    AGENT_ID: str
    AGENT_PHONE_NUMBER_ID: str
    # Add Salesforce credentials
    # SF_CLIENT_ID: str
    # SF_CLIENT_SECRET: str
    # SF_USERNAME: str
    # SF_PASSWORD: str
    # SF_SECURITY_TOKEN: str 
    # SF_INSTANCE_URL: str
    # Adding GROQ API
    GROQ_API_KEY: str
    TWILIO_AUTH_TOKEN: str
    TWILIO_ACCOUNT_SID: str
    APPWRITE_ENDPOINT: str
    APPWRITE_PROJECT_ID: str
    APPWRITE_API_KEY: str
    APPWRITE_DATABASE_ID: str
    APPWRITE_COLLECTION_ID: str

    VAPI_BASE_URL :str
    WEBHOOK_BASE_URL : str  # Optional, defaults to this
    VAPI_API_KEY :str
    VAPI_ASSISTANT_ID :str # Or uses AGENT_ID as fallback
    VAPI_PHONE_NUMBER_ID :str

    class Config:
        env_file = ".env"

settings = Settings()