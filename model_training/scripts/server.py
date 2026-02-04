import os
import json
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import edge_tts
import asyncio
import uuid
import traceback
import sqlite3
from fastapi.middleware.cors import CORSMiddleware
import re

# Configuration
DB_DIR = "../vector_store"
EXAMPLES_DB_DIR = "../vector_store_examples"
SQLITE_PATH = "aureeq.db"
MODEL_NAME = "llama3.1:8b"  # Upgraded from phi3 for better instruction following

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_menu_to_json(text):
    """Parses markdown menu into a structured JSON tree."""
    lines = text.split('\n')
    menu = {}
    current_section = None
    current_subsection = None
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Level 2 Header: Main Section (e.g. ## Starters)
        if line.startswith('## '):
            section_name = line.replace('## ', '').strip()
            current_section = section_name
            menu[current_section] = {}
            current_subsection = None # Reset subsection
            
        # Level 3 Header: Sub-section (e.g. ### Cold Mezze)
        elif line.startswith('### '):
            if current_section:
                subsection_name = line.replace('### ', '').strip()
                current_subsection = subsection_name
                menu[current_section][current_subsection] = []
                
        # List Item: Dish (e.g. * **Dish Name:** Desc)
        elif line.startswith('*'):
            item_text = line.replace('*', '').strip()
            # If we are in a subsection, add to it
            if current_section and current_subsection:
                 menu[current_section][current_subsection].append(item_text)
            # If we are just in a section (no subsection), add directly to section (using "General" or similar if needed, or simple list)
            elif current_section:
                # Handle direct items under main section if any (e.g. Baked Meat)
                if "Items" not in menu[current_section]:
                    menu[current_section]["Items"] = []
                menu[current_section]["Items"].append(item_text)
                
    return json.dumps(menu, indent=2, ensure_ascii=False)

# 1. Setup Brain (Menu Injection - Linked JSON Structure)
print("Loading Menu Data...")
try:
    MENU_PATH = os.path.join(os.path.dirname(__file__), "../data/carnivore_menu.txt")
    with open(MENU_PATH, "r", encoding="utf-8") as f:
        raw_content = f.read()
        FULL_MENU_CONTEXT = parse_menu_to_json(raw_content)
    print("Full Menu Parsed to JSON Successfully!")
except Exception as e:
    print(f"Error loading menu file: {e}")
    FULL_MENU_CONTEXT = "Menu data unavailable."
    
# Load Ingredients Data
print("Loading Ingredients Data...")
try:
    INGREDIENTS_PATH = os.path.join(os.path.dirname(__file__), "../data/ingredients.txt")
    if os.path.exists(INGREDIENTS_PATH):
        with open(INGREDIENTS_PATH, "r", encoding="utf-8") as f:
            content = f.read()
            # Simple cleanup for ingredients
            content = re.sub(r'\n\s*\n', '\n\n', content)
            INGREDIENTS_CONTEXT = content
        FULL_MENU_CONTEXT += "\n\nINGREDIENTS DATA:\n" + INGREDIENTS_CONTEXT
        print("Ingredients Loaded Successfully!")
    else:
        print("Ingredients file not found. Skipping.")
except Exception as e:
    print(f"Error loading ingredients file: {e}")

# Load Sales Examples Vector Store
print("Loading Sales Examples Vector Store...")
try:
    ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=ollama_base_url
    )
    if os.path.exists(EXAMPLES_DB_DIR):
        example_store = Chroma(persist_directory=EXAMPLES_DB_DIR, embedding_function=embeddings)
        print(f"Sales Examples Store Loaded (Ollama: {ollama_base_url})")
    else:
        print("Sales Examples Store NOT FOUND. Run ingest_examples.py first.")
        example_store = None
except Exception as e:
    print(f"Error loading Example Store: {e}")
    example_store = None

# --- Local Ollama Configuration ---

def get_llm():
    """Factory to create Local LLM instance."""
    ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return ChatOllama(
        model=MODEL_NAME, 
        base_url=ollama_base_url,
        keep_alive="24h",
        num_ctx=8192,  # Increased for full menu + RAG context
        num_predict=256,  # Increased allow for longer, more detailed responses
        temperature=0.7,  # Balanced creativity and coherence
        stop=["\n\nUser:", "USER:", "User:"]  # Stop at natural breaks to avoid mid-sentence cutoffs
    )

# Data Models
class ChatRequest(BaseModel):
    message: str
    user_id: str = None
    user_metadata: dict = None
    context: str = None # Frontend can pass pre-assembled context

# 3. Database Management
def get_db_connection():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def sync_user(email: str, name: str = None, preferences: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            if name or preferences:
                cursor.execute(
                    "UPDATE users SET name = COALESCE(?, name), preferences = COALESCE(?, preferences) WHERE email = ?",
                    (name, preferences, email)
                )
        else:
            cursor.execute(
                "INSERT INTO users (email, name, preferences) VALUES (?, ?, ?)",
                (email, name or "Guest", preferences)
            )
        conn.commit()
    except Exception as e:
        print(f"DB Error (sync_user): {e}")
    finally:
        conn.close()

def save_order(user_id: str, items: list, total_price: float):
    conn = get_db_connection()
    cursor = conn.cursor()
    order_id = str(uuid.uuid4())[:8]
    try:
        current_time = "datetime('now')" # SQLite function equivalent
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute(
            "INSERT INTO orders (id, user_id, items, total_price, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (order_id, user_id, json.dumps(items), total_price, "Completed", timestamp)
        )
        conn.commit()
        return order_id
    except Exception as e:
        print(f"DB Error (save_order): {e}")
        try:
            cursor.execute(
                "INSERT INTO orders (id, user_id, items, total_price, status) VALUES (?, ?, ?, ?, ?)",
                (order_id, user_id, json.dumps(items), total_price, "Completed")
            )
            conn.commit()
            return order_id
        except Exception:
            return None
    finally:
        conn.close()

# --- New Endpoints for Smart Frontend ---

@app.post("/api/products/search")
async def search_products(request: dict = Body(...)):
    # Simple keyword search on the full menu text since RAG is bypassed
    query = request.get("query", "").lower()
    if not query:
        return {"results": []}
    
    # Basic text search simulation
    results = []
    lines = FULL_MENU_CONTEXT.split('\n')
    for line in lines:
        if query in line.lower() and ('£' in line or '€' in line or '$' in line):
             results.append({"content": line, "metadata": {}})
    
    return {"results": results[:5]}

@app.post("/api/memory/search")
async def search_memory(request: dict = Body(...)):
    return {"results": []}

@app.get("/api/dataHandler")
async def data_handler(type: str, user_id: str = None):
    if type == "orders" and user_id:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT items, created_at, total_price FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT 5", (user_id,))
            orders = cursor.fetchall()
            return {"orders": [dict(o) for o in orders]}
        except Exception as e:
            print(f"Data Handler Error: {e}")
            return {"orders": []}
        finally:
            conn.close()
    return {"error": "Invalid request"}

# --- Chat Logic ---

SYSTEM_PROMPT_TEMPLATE = """You are Aureeq. You work for IYI restaurant.

CRITICAL RULES - FOLLOW EXACTLY:
1. Restaurant name: ONLY say "IYI" - never "IYI Dining", "Kemat Consulting", "Izmir Delights"
2. Copy dish names, prices, descriptions EXACTLY from MENU DATA - word for word
3. NEVER invent or hallucinate dishes, prices, or descriptions
4. If user asks for something not on menu, say: "IYI does not offer that, but you can try our [suggest similar dish]"
5. Response format: Copy the EXAMPLES below EXACTLY

EXAMPLES - COPY THIS EXACT FORMAT:
{examples}

MENU DATA (Your ONLY source - copy EXACTLY):
{context}

USER INFO:
{user_info}

REMEMBER: 
- Copy example format EXACTLY
- Use ONLY dishes from MENU DATA above
- Copy dish name, price (£X.XX), and description word-for-word
- Never change or invent anything
- NO MARKDOWN: Do NOT use **bold**, *italics*, or any other markdown formatting. Use plain text only."""

def get_relevant_examples(query: str, k: int = 3):
    """Retrieve top k similar examples for the query."""
    if not example_store:
        return ""
    
    try:
        # Similarity search returns Document objects
        results = example_store.similarity_search(query, k=k)
        formatted_examples = ""
        for i, doc in enumerate(results):
             # We stored "full_example" in metadata during ingestion
             # If metadata is missing, fallback to page_content (which is just user query)
             ex_text = doc.metadata.get("full_example", doc.page_content)
             formatted_examples += f"{ex_text}\n\n"
        return formatted_examples.strip()
    except Exception as e:
        print(f"Example retrieval failed: {e}")
        traceback.print_exc()
        return ""

async def generate_response(prompt_messages, user_input_text):
    """Generate response using Local Ollama."""
    
    # --- DETERMINISTIC GUARDRAILS ---
    user_query_lower = user_input_text.lower()
    
    BANNED_ITEMS = {
        "sushi": "our signature Lamb Chops",
        "pizza": "our traditional Lahmacun (Turkish Pizza)",
        "burger": "our Chicken Adana Kebab",
        "fries": "our seasoned Rice or Bread",
        "chocolate": "our handmade Baklava",
        "pasta": "our rich Bannu Pulao"
    }

    for item, pivot in BANNED_ITEMS.items():
        if item in user_query_lower:
            print(f"GUARDRAIL TRIGGERED: {item}")
            yield f"IYI does not offer it yet. However, I recommend trying {pivot} instead! Would you be interested in exploring our menu further?"
            return

    try:
        ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        print(f"Connecting to Ollama at: {ollama_base_url} with model: {MODEL_NAME}")
        llm_instance = get_llm()
        
        # Buffer to ensure sentence completion
        buffer = ""
        sentence_endings = ('.', '!', '?', '."', '!"', '?"')
        
        async for chunk in llm_instance.astream(prompt_messages):
            content = chunk.content
            buffer += content
            
            # Check if we have a complete sentence
            if any(buffer.rstrip().endswith(ending) for ending in sentence_endings):
                yield buffer
                buffer = ""
            # If buffer is getting long without sentence end, yield it anyway
            elif len(buffer) > 200:
                yield buffer
                buffer = ""
        
        # Yield any remaining content
        if buffer:
            yield buffer
                 
    except Exception as e:
        print(f"CRITICAL Model generation failed: {e}")
        traceback.print_exc()
        yield "I apologize, but I am currently experiencing technical difficulties. (Root cause: Connection to AI model failed)"

        yield "I apologize, but I am currently experiencing technical difficulties."


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.message
    provided_context = FULL_MENU_CONTEXT

    user_info_str = ""
    if request.user_metadata:
        name = request.user_metadata.get("name", "Guest")
        email = request.user_metadata.get("email", "")
        if email:
             sync_user(email, name)
        
        user_info_str += f"Name: {name}\n"
        user_info_str += f"Email: {email}\n"
        user_info_str += f"Preferences: {request.user_metadata.get('preferences', '')}\n"

    print("Retrieving relevant examples...")
    relevant_examples = get_relevant_examples(user_query, k=3)
    print(f"DEBUG: Retrieved Examples:\n{relevant_examples[:200]}...") # Log first 200 chars
    if not relevant_examples:
        relevant_examples = "No specific examples found."


    final_system = SYSTEM_PROMPT_TEMPLATE.replace("{context}", provided_context)\
                                         .replace("{user_info}", user_info_str)\
                                         .replace("{examples}", relevant_examples)
                                         
    combined_prompt = f"{final_system}\n\nUSER MESSAGE: {user_query}"
    prompt_messages = [("user", combined_prompt)]

    print(f"--- Chat Request ---")
    print(f"User: {request.user_id}")
    print(f"Prompt length: {len(combined_prompt)}")
    
    # Stream the response content directly to the client
    return StreamingResponse(generate_response(prompt_messages, user_query), media_type="text/plain")

# --- Other Endpoints ---

@app.get("/api/welcome")
async def welcome_endpoint(user_id: str = None, name: str = None):
    welcome_text = "Hello I am AUREEQ your personal assistant, How may I help you today?"
    if user_id and name:
        sync_user(user_id, name)
        
    audio_url = None
    try:
        audio_filename = f"welcome_{uuid.uuid4()}.mp3"
        audio_path = os.path.join(os.path.dirname(__file__), "../data", audio_filename)
        communicate = edge_tts.Communicate(welcome_text, "en-US-GuyNeural")
        await communicate.save(audio_path)
        audio_url = f"/audio/{audio_filename}"
    except Exception as e:
        print(f"Welcome Audio Generation Failed ({e})")
        
    return {"response": welcome_text, "audio_url": audio_url}

@app.post("/api/tts")
async def tts_endpoint(text: str = Body(..., embed=True)):
    try:
        # Simple cleanup
        if not text: return {"audio_url": None}
        
        audio_filename = f"tts_{uuid.uuid4()}.mp3"
        audio_path = os.path.join(os.path.dirname(__file__), "../data", audio_filename)
        communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
        await communicate.save(audio_path)
        return {"audio_url": f"/audio/{audio_filename}"}
    except Exception as e:
        print(f"TTS Failed: {e}")
        return {"audio_url": None}

@app.post("/api/order")
async def create_order(request: dict):
    user_id = request.get("user_id")
    items = request.get("items", [])
    total = request.get("total", 0.0)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
        
    order_id = save_order(user_id, items, total)
    if order_id:
        return {"status": "success", "order_id": order_id}
    return {"status": "error", "message": "Failed to save order"}

# Serve Audio Files
# Use absolute path based on script location to avoid CWD issues
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)
app.mount("/api/audio", StaticFiles(directory=DATA_DIR), name="audio")

if __name__ == "__main__":
    import uvicorn
    # Make sure to run with a port that matches frontend config
    uvicorn.run(app, host="0.0.0.0", port=8001)
