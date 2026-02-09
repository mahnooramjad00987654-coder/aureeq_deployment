import os
import json
import sys
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
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
MODEL_FAST = "phi3"
MODEL_REASONING = "llama3.2:1b" 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def log(msg):
    try:
        print(msg.encode('ascii', 'replace').decode('ascii'))
    except:
        print(msg)
    sys.stdout.flush()

def get_ollama_url():
    # Use environment variable for Ollama host (critical for Docker networking)
    url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    log(f"Configured Ollama URL: {url}")
    if '0.0.0.0' in url:
        url = url.replace('0.0.0.0', '127.0.0.1')
    if 'localhost' in url:
        url = url.replace('localhost', '127.0.0.1')
    if not url.startswith('http'):
        url = f"http://{url}"
    if ':11434' not in url:
        url = f"{url}:11434"
    return url

def parse_menu_to_json(text):
    """Parses markdown menu into a structured JSON tree."""
    lines = text.split('\n')
    menu = {}
    current_section = None
    current_subsection = None
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.startswith('## '):
            section_name = line.replace('## ', '').strip()
            current_section = section_name
            menu[current_section] = {}
            current_subsection = None
            
        elif line.startswith('### '):
            if current_section:
                subsection_name = line.replace('### ', '').strip()
                current_subsection = subsection_name
                menu[current_section][current_subsection] = []
                
        elif line.startswith('*'):
            item_text = line.replace('*', '').strip()
            if current_section and current_subsection:
                 menu[current_section][current_subsection].append(item_text)
            elif current_section:
                if "Items" not in menu[current_section]:
                    menu[current_section]["Items"] = []
                menu[current_section]["Items"].append(item_text)
                
    return json.dumps(menu, indent=2, ensure_ascii=False)

# 1. Setup Brain
log("Loading Menu Data...")
try:
    MENU_PATH = os.path.join(os.path.dirname(__file__), "../data/carnivore_menu.txt")
    with open(MENU_PATH, "r", encoding="utf-8") as f:
        FULL_MENU_CONTEXT = f.read()[:4000] # Cap context for speed
    log("Full Menu Loaded (Markdown, Capped)!")
except Exception as e:
    log(f"Error loading menu: {e}")
    FULL_MENU_CONTEXT = "Menu data unavailable."

# Load Ingredients Data
log("Loading Ingredients Data...")
try:
    INGREDIENTS_PATH = os.path.join(os.path.dirname(__file__), "../data/ingredients.txt")
    if os.path.exists(INGREDIENTS_PATH):
        with open(INGREDIENTS_PATH, "r", encoding="utf-8") as f:
            content = f.read()
            content = re.sub(r'\n\s*\n', '\n\n', content)
            FULL_MENU_CONTEXT += "\n\nINGREDIENTS DATA:\n" + content
        log("Ingredients Loaded Successfully!")
except Exception as e:
    log(f"Error loading ingredients: {e}")

# Load Sales Examples Vector Store
log("Loading Sales Examples Vector Store...")
try:
    ollama_base_url = get_ollama_url()
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url)
    if os.path.exists(EXAMPLES_DB_DIR):
        example_store = Chroma(persist_directory=EXAMPLES_DB_DIR, embedding_function=embeddings)
        log(f"Sales Examples Store Loaded")
    else:
        log("Sales Examples Store NOT FOUND.")
        example_store = None
except Exception as e:
    log(f"Error loading Example Store: {e}")
    example_store = None

# Startup Connectivity Check
async def verify_models(retries=5):
    ollama_url = get_ollama_url()
    import httpx
    log(f"Checking Ollama at {ollama_url}")
    for i in range(retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(ollama_url)
                if resp.status_code == 200:
                    log("✅ Ollama Service: Online")
                    list_resp = await client.get(f"{ollama_url}/api/tags")
                    models = [m['name'] for m in list_resp.json().get('models', [])]
                    for required in [MODEL_FAST, MODEL_REASONING, "nomic-embed-text:latest"]:
                        found = False
                        for m in models:
                            if required.split(':')[0] in m:
                                found = True
                                break
                        if not found:
                             log(f"⚠️ Model MISSING: {required}")
                             # Note: pulling is background task
                        else:
                            log(f"✅ Model READY: {required}")
                    return True
        except Exception as e:
            log(f"❌ Connection Attempt {i+1} Failed: {str(e)[:50]}")
            await asyncio.sleep(5)
    return False

@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(verify_models())

def get_llm(model_name):
    ollama_base_url = get_ollama_url()
    return ChatOllama(
        model=model_name, 
        base_url=ollama_base_url,
        keep_alive="24h",
        timeout=600,
        num_ctx=4096,
        temperature=0.3,
        num_thread=8
    )

class ChatRequest(BaseModel):
    message: str
    user_id: str = None
    user_metadata: dict = None
    context: str = None

def get_db_connection():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            name TEXT,
            preferences TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            items TEXT,
            total_price REAL,
            status TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def sync_user(email: str, name: str = None, preferences: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            cursor.execute("UPDATE users SET name = COALESCE(?, name), preferences = COALESCE(?, preferences) WHERE email = ?", (name, preferences, email))
        else:
            cursor.execute("INSERT INTO users (email, name, preferences) VALUES (?, ?, ?)", (email, name or "Guest", preferences))
        conn.commit()
    except Exception as e: log(f"DB Error: {e}")
    finally: conn.close()

def save_order(user_id: str, items: list, total_price: float):
    conn = get_db_connection()
    cursor = conn.cursor()
    order_id = str(uuid.uuid4())[:8]
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO orders (id, user_id, items, total_price, status, created_at) VALUES (?, ?, ?, ?, ?, ?)", (order_id, user_id, json.dumps(items), total_price, "Completed", timestamp))
        conn.commit()
        return order_id
    except Exception as e:
        log(f"DB Error (save_order): {e}")
        return None
    finally: conn.close()

# --- Chat Logic ---

# --- Router Logic ---

PROMPT_FAST = """You are Aureeq, the personal food assistant for IYI restaurant.
INTERNAL CONFIGURATION: Handle SIMPLE, DIRECT queries.
STRICT RULES:
1. MENU ONLY: If item is not in menu, say: "Sorry, we don't offer that item at the moment, but you can explore our other options."
2. EXACTNESS: Use exact names and prices from menu. No inventing.
3. BREVITY: Max 2-4 sentences. Friendly, human tone.
4. ORDER TAGGING: ONLY append [ORDER: Item Name | Price] if user EXPLICITLY says "add", "buy", or "order".
   - IF user asks "What is X?" -> NO TAG.
   - IF user says "I like X" -> NO TAG.
   - IF you suggest X -> NO TAG.
   - ONLY when user says "Add X to cart" -> TAG.
5. NON-FOOD: If query is non-food, say: "I apologize, but I am a food assistant and cannot help with that. However, I can help you find delicious items on the IYI menu! Would you like a suggestion?"
6. NO CHATTER: Do NOT state your internal role or limitations (e.g., "I handle simple queries") in normal conversation. Just answer the user.
7. CONTEXT: Name: {user_name}. Preferences: {user_preferences}.

MENU DATA:
{context}
"""

PROMPT_REASONING = """You are Aureeq, the personal food assistant for IYI restaurant.
INTERNAL CONFIGURATION: Provide recommendations.
STRICT RULES:
1. MENU ONLY: Use menu data. Do not invent items.
2. RECOMMEND: Suggest ONE item based on user request.
3. STYLE: Use the provided EXAMPLE as style guidance only. Do NOT copy it.
4. BREVITY: Max 2-4 sentences. No long explanations.
5. NO ORDER TAGS: Do NOT generate [ORDER: ...] tags. Ask for confirmation first.
6. NON-FOOD: If query is non-food, say: "I apologize, but I am a food assistant and cannot help with that. However, I can help you find delicious items on the IYI menu! Would you like a suggestion?"
7. NO CHATTER: Do NOT state your internal role or limitations in normal conversation.
8. CONTEXT: Name: {user_name}. Preferences: {user_preferences}.

MENU DATA:
{context}

EXAMPLE STYLE (Do not copy contents, only tone):
{examples}
"""

def route_query(text):
    text = text.lower()
    
    # 1. Guardrails (Non-food) - Return "reject"
    non_food = ["weather", "news", "coding", "programming", "politics", "joke", "sports", "movie", "technology", "stock", "finance"]
    if any(w in text for w in non_food):
        return "reject"

    # 2. Reasoning / Recommendations -> Llama
    reasoning_keywords = [
        "recommend", "suggest", "best", "what should i", "choose",
        "sweet", "spicy", "light", "heavy", "drink", "dessert", "main", "something"
    ]
    if any(w in text for w in reasoning_keywords):
        return "reasoning"

    # 3. Default -> Fast (Greetings, Menu, Order, Hunger)
    return "fast"

async def get_relevant_examples_async(query: str, k: int = 3):
    if not example_store: return ""
    try:
        results = await asyncio.to_thread(example_store.similarity_search, query, k=k)
        return "\n\n".join([doc.metadata.get("full_example", doc.page_content) for doc in results]).strip()
    except Exception as e:
        log(f"ERROR RAG: {e}")
        return ""

# --- API Endpoints ---

@app.post("/api/products/search")
async def search_products(payload: dict = Body(...)):
    query = payload.get("query", "").lower()
    log(f"DEBUG: Product search: {query}")
    results = []
    lines = FULL_MENU_CONTEXT.replace('{', '').replace('}', '').replace('"', '').split('\n')
    for line in lines:
        if query in line.lower() and ('£' in line or '€' in line or ':' in line):
             results.append({"content": line.strip(), "metadata": {}})
    return {"results": results[:5]}

@app.post("/api/memory/search")
async def search_memory(payload: dict = Body(...)):
    return {"results": []}

@app.get("/api/dataHandler")
async def data_handler(type: str, user_id: str = None):
    log(f"DEBUG: Data handler: {type} for {user_id}")
    if type == "orders" and user_id:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT items, created_at, total_price FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT 5", (user_id,))
            orders = cursor.fetchall()
            order_list = [dict(o) for o in orders]
            return {
                "results": [{"content": f"Past order: {o['items']}", "metadata": {}} for o in order_list],
                "orders": order_list
            }
        finally: conn.close()
    return {"results": [], "orders": []}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.message
    log(f"--- Chat Request: {user_query} ---")

    async def chat_generator():
        # Yield Zero-Width Space to keep connection open without visible text
        yield "\u200B"
        
        try:
            # 1. User Context
            name = "Guest"
            preferences = ""
            if request.user_metadata:
                name = request.user_metadata.get("name", "Guest")
                preferences = request.user_metadata.get("preferences", "")
                if request.user_metadata.get("email"): 
                    sync_user(request.user_metadata["email"], name, preferences)

            # 2. Routing
            route = route_query(user_query)
            log(f"DEBUG: Router decided: {route.upper()}")

            if route == "reject":
                yield "I am only trained to help you with food and menu selections."
                return

            model_to_use = MODEL_FAST
            prompt = ""
            
            if route == "fast":
                model_to_use = MODEL_FAST
                system = PROMPT_FAST.replace("{context}", FULL_MENU_CONTEXT)\
                                    .replace("{user_name}", name)\
                                    .replace("{user_preferences}", preferences)
                prompt = f"{system}\n\nUser: {user_query}\nAureeq: "
            
            else: # reasoning
                model_to_use = MODEL_REASONING
                # Fetch MAX 1 example
                example_text = await get_relevant_examples_async(user_query, k=1)
                system = PROMPT_REASONING.replace("{context}", FULL_MENU_CONTEXT)\
                                         .replace("{user_name}", name)\
                                         .replace("{user_preferences}", preferences)\
                                         .replace("{examples}", example_text or "No examples.")
                prompt = f"{system}\n\nUser: {user_query}\nAureeq: "

            # 3. Generation
            log(f"DEBUG: Invoking {model_to_use}...")
            # ollama_url = get_ollama_url()
            ollama_url = "http://127.0.0.1:11434"
            import httpx
            
            async with httpx.AsyncClient(timeout=600.0, trust_env=False) as client:
                async with client.stream("POST", f"{ollama_url}/api/generate", json={
                    "model": model_to_use,
                    "prompt": prompt,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": "24h",
                    "options": {"temperature": 0.3}
                }) as response:
                    if response.status_code != 200:
                        yield f"[Error: {response.status_code}]"
                        return

                    async for line in response.aiter_lines():
                        if not line: continue
                        try:
                            chunk = json.loads(line)
                            if "error" in chunk: break
                            content = chunk.get("response")
                            if content: yield content
                            if chunk.get("done"): break
                        except Exception as e:
                            log(f"Chunk parsing error: {e} in line: {line}")
        except Exception as e:
            log(f"CRITICAL ERROR: {e}")
            yield f"[Error: {str(e)[:40]}]"

    return StreamingResponse(
        chat_generator(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/plain; charset=utf-8"
        }
    )

@app.get("/api/welcome")
async def welcome_endpoint(user_id: str = None, name: str = None):
    welcome_text = "Hello I am AUREEQ your personal assistant, How may I help you today?"
    audio_filename = f"welcome_{uuid.uuid4()}.mp3"
    audio_path = os.path.join(DATA_DIR, audio_filename)
    try:
        communicate = edge_tts.Communicate(welcome_text, "en-US-GuyNeural")
        await communicate.save(audio_path)
        return {"response": welcome_text, "audio_url": f"/audio/{audio_filename}"}
    except Exception:
        return {"response": welcome_text, "audio_url": None}

@app.post("/api/tts")
async def tts_endpoint(text: str = Body(..., embed=True)):
    audio_filename = f"tts_{uuid.uuid4()}.mp3"
    audio_path = os.path.join(DATA_DIR, audio_filename)
    try:
        communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
        await communicate.save(audio_path)
        return {"audio_url": f"/audio/{audio_filename}"}
    except Exception:
        return {"audio_url": None}

@app.post("/api/order")
async def create_order(request: dict):
    order_id = save_order(request.get("user_id"), request.get("items", []), request.get("total", 0.0))
    if order_id: return {"status": "success", "order_id": order_id}
    raise HTTPException(status_code=500, detail="Failed to save order")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)
app.mount("/api/audio", StaticFiles(directory=DATA_DIR), name="audio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
