import os
from dotenv import load_dotenv

# Load environment variables (works locally)
load_dotenv()

# API Keys - works both locally and on Hugging Face
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Settings
GROQ_MODEL = "llama-3.3-70b-versatile"

# Paths
DATA_DIR = "data/"
EMBEDDINGS_DIR = "embeddings/"

# Vector DB Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Validate API Key
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found! Please check your .env file or Space secrets.")
else:
    print("✅ Config loaded successfully!")