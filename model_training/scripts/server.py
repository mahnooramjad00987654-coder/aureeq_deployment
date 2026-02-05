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
MODEL_NAME = "llama3.1:8b" 

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
print("Loading Menu Data...")
try:
    MENU_PATH = os.path.join(os.path.dirname(__file__), "../data/carnivore_menu.txt")
    with open(MENU_PATH, "r", encoding="utf-8") as f:
        raw_content = f.read()
        FULL_MENU_CONTEXT = parse_menu_to_json(raw_content)
    print("Full Menu Parsed Successfully!")
except Exception as e:
    print(f"Error loading menu: {e}")
    FULL_MENU_CONTEXT = "Menu data unavailable."

# Load Ingredients Data
print("Loading Ingredients Data...")
try:
    INGREDIENTS_PATH = os.path.join(os.path.dirname(__file__), "../data/ingredients.txt")
    if os.path.exists(INGREDIENTS_PATH):
        with open(INGREDIENTS_PATH, "r", encoding="utf-8") as f:
            content = f.read()
            content = re.sub(r'\n\s*\n', '\n\n', content)
            FULL_MENU_CONTEXT += "\n\nINGREDIENTS DATA:\n" + content
        print("Ingredients Loaded Successfully!")
except Exception as e:
    print(f"Error loading ingredients: {e}")

# Load Sales Examples Vector Store
print("Loading Sales Examples Vector Store...")
try:
    ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url)
    if os.path.exists(EXAMPLES_DB_DIR):
        example_store = Chroma(persist_directory=EXAMPLES_DB_DIR, embedding_function=embeddings)
        print(f"Sales Examples Store Loaded")
    else:
        print("Sales Examples Store NOT FOUND.")
        example_store = None
except Exception as e:
    print(f"Error loading Example Store: {e}")
    example_store = None

# Startup Connectivity Check
async def verify_models(retries=5):
    ollama_url = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    import httpx
    print(f"Checking Ollama at {ollama_url}")
    for i in range(retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(ollama_url)
                if resp.status_code == 200:
                    print("✅ Ollama Service: Online")
                    list_resp = await client.get(f"{ollama_url}/api/tags")
                    models = [m['name'] for m in list_resp.json().get('models', [])]
                    for required in [MODEL_NAME, "nomic-embed-text:latest"]:
                        if not any(required.split(':')[0] in m for m in models):
                            print(f"⚠️ Model MISSING: {required}")
                            await client.post(f"{ollama_url}/api/pull", json={"name": required}, timeout=1.0)
                        else:
                            print(f"✅ Model READY: {required}")
                    return True
        except Exception as e:
            print(f"❌ Connection Attempt {i+1} Failed: {str(e)[:50]}")
            await asyncio.sleep(5)
    return False

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(verify_models())

def get_llm():
    ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return ChatOllama(
        model=MODEL_NAME, 
        base_url=ollama_base_url,
        keep_alive="24h",
        timeout=600,
        num_ctx=4096,
        temperature=0.3,
        num_thread=8,
        stop=["\n\nUser:", "USER:", "User:"]
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
    except Exception as e: print(f"DB Error: {e}")
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
        print(f"DB Error (save_order): {e}")
        return None
    finally: conn.close()

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
9. ORDERING: Do NOT append an order tag during initial recommendations. ONLY append the [ORDER: Exact Dish Name | Price] tag when the user explicitly: (a) asks to 'add to cart', (b) says 'lets continue with [dish]', or (c) says 'yes' to your offer of adding that specific dish to their cart. You may ask 'Would you like to add this [dish] to your cart?' if they seem interested, but only show the tag AFTER they confirm with a 'yes'.
10. NO MARKDOWN: Use plain text only. Do NOT use **bold** or *italics*.

EXAMPLES - COPY THIS EXACT FORMAT:
{examples}

MENU DATA (Your ONLY source - copy EXACTLY):
{context}

USER INFO:
{user_info}

REMEMBER: No casual apologies. Be formal, always recommend, and guide the guest to IYI's offerings."""

async def get_relevant_examples_async(query: str, k: int = 3):
    if not example_store: return ""
    try:
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
             try:
                 results = await asyncio.wait_for(loop.run_in_executor(pool, lambda: example_store.similarity_search(query, k=k)), timeout=10.0)
                 return "\n\n".join([doc.metadata.get("full_example", doc.page_content) for doc in results]).strip()
             except asyncio.TimeoutError:
                 print("⚠️ RAG Timeout")
                 return ""
    except Exception as e:
        print(f"ERROR RAG: {e}")
        return ""

# --- API Endpoints ---

@app.post("/api/products/search")
async def search_products(request: dict = Body(...)):
    query = request.get("query", "").lower()
    if not query: return {"results": []}
    
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
        finally: conn.close()
    return {"error": "Invalid request"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.message
    print(f"--- Chat Request: {user_query[:50]}... ---")

    async def chat_generator():
        yield " " 
        try:
            name, email, user_info_str = "Guest", "", ""
            if request.user_metadata:
                name = request.user_metadata.get("name", "Guest")
                email = request.user_metadata.get("email", "")
                if email: sync_user(email, name)
                user_info_str = f"Name: {name}\nEmail: {email}\nPreferences: {request.user_metadata.get('preferences', '')}\n"

            relevant_examples = await get_relevant_examples_async(user_query, k=3)
            yield " " 

            user_query_lower = user_query.lower()
            NON_FOOD_KEYWORDS = ["weather", "news", "coding", "programming", "joke", "politics"]
            if any(word in user_query_lower for word in NON_FOOD_KEYWORDS):
                 yield "I can't assist you with this. However, I highly recommend our Lamb Chops from the IYI menu."
                 return

            BANNED_ITEMS = {"sushi": "Lamb Chops", "pizza": "Lahmacun", "burger": "Chicken Adana"}
            for item, pivot in BANNED_ITEMS.items():
                if item in user_query_lower:
                    yield f"IYI doesn't offer {item} right now but you can have other options from our menu. I recommend our {pivot}!"
                    return

            final_system = SYSTEM_PROMPT_TEMPLATE.replace("{context}", FULL_MENU_CONTEXT)\
                                               .replace("{examples}", relevant_examples or "No examples available.")\
                                               .replace("{user_info}", user_info_str)
            
            prompt_messages = [("user", f"{final_system}\n\nUSER MESSAGE: {user_query}")]
            
            # Connection Stabilizer
            llm = get_llm()
            has_started = False
            async for chunk in llm.astream(prompt_messages):
                content = getattr(chunk, 'content', chunk) if not isinstance(chunk, str) else chunk
                if content:
                    has_started = True
                    yield content
            
            if not has_started:
                 yield "\n[System: The brain is currently initializing. Please try again.]"

        except Exception as e:
            traceback.print_exc()
            yield f"\n\n[Connectivity issue: Please check back in a moment.]"

    return StreamingResponse(
        chat_generator(), 
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/api/welcome")
async def welcome_endpoint(user_id: str = None, name: str = None):
    welcome_text = "Hello I am AUREEQ your personal assistant, How may I help you today?"
    audio_url = None
    try:
        audio_filename = f"welcome_{uuid.uuid4()}.mp3"
        audio_path = os.path.join(DATA_DIR, audio_filename)
        communicate = edge_tts.Communicate(welcome_text, "en-US-GuyNeural")
        await communicate.save(audio_path)
        audio_url = f"/audio/{audio_filename}"
    except Exception: pass
    return {"response": welcome_text, "audio_url": audio_url}

@app.post("/api/tts")
async def tts_endpoint(text: str = Body(..., embed=True)):
    try:
        audio_filename = f"tts_{uuid.uuid4()}.mp3"
        audio_path = os.path.join(DATA_DIR, audio_filename)
        communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
        await communicate.save(audio_path)
        return {"audio_url": f"/audio/{audio_filename}"}
    except Exception: return {"audio_url": None}

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
