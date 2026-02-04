import os
import json
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
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
MODEL_NAME = "llama3.2"  # Standard tag for 3B version

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

# Startup Connectivity Check & Model Verifier
async def verify_models(retries=5):
    ollama_url = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    import httpx
    
    print(f"--- Production Startup: Checking Ollama at {ollama_url} ---")
    
    for i in range(retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # 1. Wait for Ollama service
                resp = await client.get(ollama_url)
                if resp.status_code != 200:
                    raise Exception(f"Service returned {resp.status_code}")
                
                print("✅ Ollama Service: Online")
                
                # 2. Check for required models
                list_resp = await client.get(f"{ollama_url}/api/tags")
                models = [m['name'] for m in list_resp.json().get('models', [])]
                
                for required in [MODEL_NAME, "nomic-embed-text:latest"]:
                    # Clean model name for comparison (tags often add :latest)
                    clean_req = required if ":" in required else f"{required}:latest"
                    if not any(m.startswith(required.split(':')[0]) for m in models):
                        print(f"⚠️ Model MISSING: {required}. Attempting background pull...")
                        # Fire and forget pull
                        await client.post(f"{ollama_url}/api/pull", json={"name": required}, timeout=1.0)
                    else:
                        print(f"✅ Model READY: {required}")
                
                return True
        except Exception as e:
            print(f"❌ Connection Attempt {i+1} Failed: {str(e)[:100]}")
            if i < retries - 1:
                await asyncio.sleep(5)
    return False

@app.on_event("startup")
async def startup_event():
    # Run in background to not block startup
    asyncio.create_task(verify_models())

def get_llm():
    """Factory to create Local LLM instance."""
    ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return ChatOllama(
        model=MODEL_NAME, 
        base_url=ollama_base_url,
        keep_alive="24h",
        timeout=120, # Reduced timeout as 3b is much faster
        num_ctx=4096, # Reduced context window for speed (more than enough for menu)
        temperature=0.7,
        num_thread=4, # Optimize for parallel CPU execution
        stop=["\n\nUser:", "USER:", "User:"]
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

SYSTEM_PROMPT_TEMPLATE = """You are Aureeq, the formal and confident personal assistant for IYI restaurant.

STRICT COMPLIANCE RULES - FOLLOW EXACTLY:
1. TONE: Be formal, confident, and precise. Avoid being overly apologetic.
2. GREETINGS: Always respond with a formal "Hello" or "Hi" when the user greets you.
3. RECOMMENDATIONS: Always recommend a specific item from the IYI menu in every response to guide the user's choice.
4. APOLOGIES: NEVER say "sorry" or "I apologize" for normal conversation. ONLY use a polite refusal for non-food or off-menu requests as specified below.
5. NON-FOOD ITEMS: If the user asks for anything not related to food or IYI, say: "I can't assist you with this. However, I highly recommend our [Dish Name] from the IYI menu."
6. OFF-MENU FOOD: If the user asks for a food item not found in the MENU DATA, say: "IYI doesn't offer it right now but you can have other options from our menu," and then immediately recommend a similar item from our actual menu.
7. RESTAURANT NAME: Only say "IYI" - never "IYI Dining", "Kemat Consulting", or "Izmir Delights".
8. MENU DATA: Copy dish names, prices, and descriptions EXACTLY from the provided MENU DATA - word for word.
9. ORDERING: When a user expresses a clear intent to buy or order a specific dish, you MUST append the following tag at the end of your response: [ORDER: Exact Dish Name | Price].
10. NO MARKDOWN: Use plain text only. Do NOT use **bold** or *italics*.

EXAMPLES - COPY THIS EXACT FORMAT:
{examples}

MENU DATA (Your ONLY source - copy EXACTLY):
{context}

USER INFO:
{user_info}

REMEMBER: No casual apologies. Be formal, always recommend, and guide the guest to IYI's offerings."""

async def get_relevant_examples_async(query: str, k: int = 3):
    """Retrieve top k similar examples with a strict timeout."""
    if not example_store:
        return ""
    
    try:
        # Run the blocking similarity search in a thread with a timeout
        print(f"DEBUG: Starting similarity search (async wrapper)...")
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
             # Timeout of 10 seconds for vector search
             try:
                 results = await asyncio.wait_for(
                     loop.run_in_executor(pool, lambda: example_store.similarity_search(query, k=k)),
                     timeout=10.0
                 )
                 print(f"DEBUG: Similarity search finished. Results: {len(results)}")
                 formatted_examples = ""
                 for i, doc in enumerate(results):
                      ex_text = doc.metadata.get("full_example", doc.page_content)
                      formatted_examples += f"{ex_text}\n\n"
                 return formatted_examples.strip()
             except asyncio.TimeoutError:
                 print("⚠️ RAG Search TIMEOUT - Skipping examples to avoid hang.")
                 return ""
    except Exception as e:
        print(f"ERROR: Example retrieval failed: {e}")
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

    # Hardcoded check for non-food queries to ensure compliance
    NON_FOOD_KEYWORDS = ["weather", "news", "code", "programming", "joke", "story", "politics", "crypto", "stock"]
    for word in NON_FOOD_KEYWORDS:
        if word in user_query_lower:
             yield "Sorry, I am only trained to assist you in your food selection."
             return

    # Check for common off-menu items
    for item, pivot in BANNED_ITEMS.items():
        if item in user_query_lower:
            print(f"GUARDRAIL TRIGGERED: {item}")
            yield f"Sorry, IYI does not offer {item} yet. However, I recommend trying {pivot} instead! Would you be interested in exploring our menu further?"
            return

    try:
        ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        print(f"Connecting to Ollama at: {ollama_base_url} with model: {MODEL_NAME}")
        llm_instance = get_llm()
        
        # Use a timeout context if possible, or just rely on astreams's internal handling
        async for chunk in llm_instance.astream(prompt_messages):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
            elif isinstance(chunk, str):
                yield chunk
                 
    except Exception as e:
        print(f"CRITICAL Model generation failed: {e}")
        traceback.print_exc()
        # Only yield if we haven't already started sending the response
        # In a StreamingResponse, this might still show up.
        yield "\n\n[Connectivity issue: Please try again in a moment while the AI initializes.]"


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

    print(f"--- Chat Request Received ---")
    print(f"Message: {user_query[:50]}...")
    
    # Stream the response content directly to the client
    async def chat_generator():
        # NOTE: Removed initial space yield to avoid empty display bubbles in frontend trim
        
        # 1. RETRIEVE EXAMPLES (with timeout safety)
        print("Retrieving relevant sales examples...")
        try:
            relevant_examples = await get_relevant_examples_async(user_query, k=3)
            if not relevant_examples:
                print("INFO: No relevant examples found or search timed out.")
                relevant_examples = "No specific examples found."
            print(f"DEBUG: Examples context ready ({len(relevant_examples)} chars)")
        except Exception as e:
            print(f"ERROR: RAG Retrieval Path Failed: {e}")
            relevant_examples = "Context unavailable."

        final_system = SYSTEM_PROMPT_TEMPLATE.replace("{context}", provided_context)\
                                           .replace("{examples}", relevant_examples)\
                                           .replace("{user_info}", user_info_str)
        
        combined_prompt = f"{final_system}\n\nUSER MESSAGE: {user_query}"
        prompt_messages = [("user", combined_prompt)]
        
        print(f"Starting LLM with prompt ({len(combined_prompt)} chars)...")
        try:
            # Check model presence before streaming to provide better error
            ollama_url = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            import httpx
            async with httpx.AsyncClient() as client:
                models_resp = await client.get(f"{ollama_url}/api/tags")
                models = [m['name'] for m in models_resp.json().get('models', [])]
                if not any(m.startswith(MODEL_NAME.split(':')[0]) for m in models):
                    yield "Aureeq is still initializing her brain (downloading models). Please wait 2-3 minutes and try again..."
                    return

            llm_instance = get_llm()
            async for chunk in llm_instance.astream(prompt_messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk
        except Exception as e:
            print(f"CRITICAL Stream Error: {e}")
            yield "\n\n[Connectivity issue: The AI is taking too long to respond. Please try a shorter message.]"

    return StreamingResponse(
        chat_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Content-Type-Options": "nosniff"
        }
    )

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
