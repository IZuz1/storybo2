import asyncio
import telegram
from telegram import Bot, Poll, Message
import json
import os
from pathlib import Path
import logging
from openai import OpenAI, OpenAIError
import random
import requests  # Для запросов к Stability Diffusion API
from typing import Optional, List, Dict, Any

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename="thecode.log")

# --- Configuration ---
# !! IMPORTANT: Replace placeholders below with your actual values !!
BOT_TOKEN = "8058100768:AAFpSm1gmmKdZallAnFG1UogRHaByY7JYe4"
CHANNEL_ID = "@testovidhe"
# Use environment variable for API key, fallback to placeholder
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")  # дефолт - OpenAI

# Stability Diffusion Configuration
SD_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image"  # Или другой endpoint
SD_API_KEY = os.getenv("STABILITY_API_KEY", "sk-yqVBIXAPOd31oyuHwaYEeb2GPI4YuQy9JB7DvY8BlUPL5DSE")  # Обязательно!
# IMAGE_STYLE перенес в настройки Stability Diffusion
IMAGE_SIZE = "1024x1024"  # Оставил, если используется где-то еще
STATE_FILE = Path(__file__).parent / "story_state.json"
POLL_QUESTION_TEMPLATE = "Как продолжится история?"

# OpenAI Settings
OPENAI_MODEL = "openai/gpt-4o"
MAX_CONTEXT_CHARS = 15000

# --- End Configuration ---

# --- OpenAI Client Initialization ---
openai_client = None
if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY":
    try:
        openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://cyberfront1945.ru",
                "X-Title": "Story Generator",
            }
        )
        logging.info("OpenAI client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
else:
    logging.warning("OPENAI_API_KEY not found or is placeholder. LLM features will be disabled.")

# --- Helper Function to Validate Configuration ---
def validate_config() -> bool:
    """Checks if the configuration values have been changed from placeholders."""
    valid = True
    if BOT_TOKEN == "YOUR_BOT_TOKEN" or not BOT_TOKEN:
        logging.error("BOT_TOKEN is not set correctly.")
        valid = False
    if not CHANNEL_ID or CHANNEL_ID == "@your_channel_username":
        logging.error("CHANNEL_ID is not set correctly.")
        valid = False
    if not openai_client:
        logging.warning("OpenAI client is not initialized. Check OPENAI_API_KEY. LLM features disabled.")
    if not INITIAL_STORY_IDEA:
        logging.error("INITIAL_STORY_IDEA cannot be empty.")
        valid = False
    if SD_API_KEY == "YOUR_STABILITY_AI_KEY" or not SD_API_KEY:
        logging.error("STABILITY_API_KEY is not set correctly. Image generation will fail.")
        valid = False # Поставил False, т.к. без ключа генерация сломается.
    return valid

# --- State Management ---
def load_state() -> Dict[str, Any]:
    """Loads the story state from the JSON file."""
    default_state = {"current_story": "", "last_poll_message_id": None}
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
                state["current_story"] = state.get("current_story", default_state["current_story"])
                state["last_poll_message_id"] = state.get("last_poll_message_id", default_state["last_poll_message_id"])
                logging.info(f"State loaded from {STATE_FILE}: {state}")
                return state
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading state file {STATE_FILE}: {e}. Starting fresh.")
            return default_state
    else:
        logging.info("State file not found. Starting fresh.")
        return default_state

def save_state(state: Dict[str, Any]) -> None:
    """Saves the story state to the JSON file."""
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
        logging.info(f"Story state saved to {STATE_FILE}: {state}")
    except IOError as e:
        logging.error(f"Error saving state file {STATE_FILE}: {e}")

# --- OpenAI Interaction Functions ---
def generate_story_continuation_openai(current_story: str, user_choice: str) -> Optional[str]:
    """Calls OpenAI API to get the next story part using strict function calling."""
    if not openai_client:
        logging.warning("OpenAI client not available. Skipping story generation.")
        return "\n\n[Продолжение не сгенерировано - OpenAI недоступен]"

    logging.info("Generating story continuation via OpenAI...")
    truncated_story = current_story
    if len(current_story) > MAX_CONTEXT_CHARS:
        logging.warning(f"Current story context ({len(current_story)} chars) exceeds limit ({MAX_CONTEXT_CHARS}). Truncating.")
        truncated_story = current_story[-MAX_CONTEXT_CHARS:]

    system_prompt = """
    Ты - самый великий современный творческий писатель, продолжающий интерактивную историю на русском языке. 
    Тебе дан предыдущий текст истории и выбор пользователя (победитель опроса), который определяет следующее направление. 

    Твоя задача - написать СЛЕДУЮЩИЕ ТРИ ПАРАГРАФА истории, органично продолжая сюжет под влиянием выбора пользователя. Каждый параграф должен быть отделен пустой строкой. 

    ###Правила напсиания###
    – Никогда не обращайся к персонажу "герой" или "героиня", давай им имя.

    – Ты прекрасно знаешь как писать интересно и креативно. Твоя задча интерактивно менять историю, в зависимости от событий в рассказе – но вся история ДОЛЖНА БЫТЬ СВЯЗНОЙ.

    – Никогда не пиши с "AI SLOP"

    – Меняй детальность истории, в зависимости от типов событий. Ниже — базовые «темпоральные правила» – «Тип события = сколько реального времени в среднем помещается в один абзац», а затем коротко — как выбор этих масштабов усиливает или снижает летальность сцены:

    <temporal>
    Фоновое описание обычного дня = ≈ 3 часа
    Диалог (реплика ↔ ответ) = ≈ 5 минут
    Битва / рукопашная схватка = ≈ 2 минуты
    Кризис без боя (погоня, взлом, спасение) = ≈ 30 минут
    Внутренний монолог / размышление = ≈ 45 минут
    Переходное «прошла неделя» = ≈ 36 часов
    Исторический дайджест, газетная вставка = ≈ 10 дней
    </temporal>


    ###Правила ответа###
    – Возвращай результат ТОЛЬКО в формате JSON, используя предоставленный инструмент 'write_story_part' с полями:
    – 'reasoning' – твои мысли о том, как ты продолжишь историю чтобы действия пользователя органично вписались, добавь туда "две банальности которые ты избежишь" что избежать клише. Не параграфа на этот пункт;
    – 'story_part' – сам текст следующих трех параграфов истории;
    Не добавляй никакого другого текста.

    Всегда следуй ###Правила напсиания### и ###Правила ответа###.
    """

    user_prompt = f"""Предыдущая история:
    {truncated_story}

    Выбор пользователя: '{user_choice}'

    Напиши следующие три параграфа, используя инструмент 'write_story_part'."""

    story_tool = {
        "type": "function",
        "function": {
            "name": "write_story_part",
            "description": "Записывает следующие три абзаца интерактивной истории и обоснование.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Краткое обоснование или план для следующих трех параграфов истории на русском языке."
                    },
                    "story_part": {
                        "type": "string",
                        "description": "Текст следующих трех параграфов истории на русском языке, разделенных пустой строкой."
                    }
                },
                "required": ["reasoning", "story_part"],
                'additionalProperties': False
            }
        }
    }

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=[story_tool],
            tool_choice={"type": "function", "function": {"name": "write_story_part"}}
        )

        logging.debug(f"Raw OpenAI response: {response}")

        if not response or not response.choices or not response.choices[0].message:
            logging.error("Invalid OpenAI response structure")
            return None

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls or not tool_calls[0].function:
            logging.error("No tool_calls or function data in OpenAI response")
            logging.debug(f"Problematic response structure: {response}")
            return None

        if tool_calls[0].function.name != "write_story_part":
            logging.error(f"Unexpected tool call name: {tool_calls[0].function.name}")
            return None

        try:
            arguments = json.loads(tool_calls[0].function.arguments)
            new_story_part = arguments.get("story_part")
            reasoning = arguments.get("reasoning")
            logging.info(f"Successfully parsed story part: {new_story_part[:100]}...")
            return new_story_part
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse function arguments: {e}")
            return None

    except OpenAIError as e:
        logging.error(f"OpenAI API error during story generation: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Unexpected error during story generation: {e}", exc_info=True)
        return None
    
async def generate_poll_options_openai(full_story_context: str) -> Optional[List[str]]:
    """Calls OpenAI API to get 4 poll options using strict function calling."""
    if not openai_client:
        logging.warning("OpenAI client not available. Skipping poll option generation.")
        return None

    logging.info("Generating poll options via OpenAI...")
    truncated_context = full_story_context[-MAX_CONTEXT_CHARS:]

    system_prompt = """Ты - помощник для интерактивной истории на русском языке. 
    Тебе дан ПОЛНЫЙ текущий текст истории. Твоя задача - придумать ровно 4 КОРОТКИХ (максимум 90 символов!) и ФУНДАМЕНТАЛЬНО РАЗНЫХ варианта продолжения сюжета для опроса в Telegram. 
    Варианты должны быть МАКСИМАЛЬНО НЕПОХОЖИМИ друг на друга, предлагая совершенно разные, возможно, даже противоположные, направления развития событий (например, пойти на север ИЛИ пойти на юг ИЛИ остаться на месте ИЛИ искать что-то конкретное).
    Избегай незначительных вариаций одного и того же действия. Нужны действительно ОТЛИЧАЮЩИЕСЯ выборы.
    Возвращай результат ТОЛЬКО в формате JSON, используя предоставленный инструмент 'suggest_poll_options' с полем 'options' (массив из 4 строк). Не добавляй никакого другого текста."""

    user_prompt = f"""Полный текст текущей истории:
    {truncated_context}

    Предложи 4 варианта для опроса, используя инструмент 'suggest_poll_options'."""

    poll_tool = {
        "type": "function",
        "function": {
            "name": "suggest_poll_options",
            "description": "Предлагает 4 варианта продолжения для опроса в интерактивной истории.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "options": {
                        "type": "array",
                        "description": "List of exactly 4 concise story continuation options (max 90 chars each) in Russian.",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["options"],
                "additionalProperties": False
            }
        }
    }

    try:
        for attempt in range(3):
            try:
                response = await openai_client.chat.completions.create( # await
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Контекст: {full_story_context[-MAX_CONTEXT_CHARS:]}"}
                    ],
                    tools=[poll_tool],
                    tool_choice={"type": "function", "function": {"name": "suggest_poll_options"}},
                    timeout=20
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logging.warning(f"Attempt {attempt+1}/3 failed: {str(e)}")
                await asyncio.sleep(1) # await
                
        if not response or not response.choices or not response.choices[0].message or not response.choices[0].message.tool_calls:
            logging.error("Invalid API response structure")
            return None

        tool_calls = response.choices[0].message.tool_calls

        try:
            args = json.loads(tool_calls[0].function.arguments)
            options = args.get('options', [])
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"JSON parsing failed: {str(e)}")
            return None

        valid_options = []
        for opt in options:
            if isinstance(opt, str) and 10 <= len(opt) <= 90:
                valid_options.append(opt.strip()[:90])
            if len(valid_options) >= 4:
                break

        if len(valid_options) != 4:
            logging.error(f"Invalid options count: {len(valid_options)}")
            return None

        logging.info(f"Generated valid options: {valid_options}")
        return valid_options

    except Exception as e:
        logging.error(f"Poll generation failed: {str(e)}")
        return None

# --- Stability Diffusion Image Generation ---
def generate_image_stability_diffusion(story_context: str, image_style: str = "pixel art") -> Optional[str]:
    """
    Generates an image using the Stability Diffusion API based on the story context.

    Args:
        story_context: The text describing the scene to be visualized.
        image_style: стиль генерации изображения.

    Returns:
        The URL of the generated image, or None on error.
    """
    if not SD_API_KEY:
        logging.warning("STABILITY_API_KEY is not set. Skipping image generation.")
        return None

    try:
        prompt = f"{image_style} of a scene from the story: {story_context[:800]}"  # Truncate context if needed
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {SD_API_KEY}",
        }
        data = {
            "model": "stable-diffusion-xl-1024-v1-0",  # Явно укажем модель
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "guidance_scale": 8,
            "negative_prompt": "blurry, distorted, ugly, incoherent",
            "samples": 1,
        }

        logging.info(f"Sending image generation request to Stability Diffusion API with prompt: {prompt[:100]}...")
        response = requests.post(SD_API_URL, headers=headers, json=data, timeout=60)  # Установим таймаут

        if response.status_code != 200:
            logging.error(f"Stability Diffusion API error: {response.status_code}, {response.text}")
            return None

        response_data = response.json()
        if "artifacts" not in response_data or not response_data["artifacts"]:
            logging.error("No artifacts found in Stability Diffusion response.")
            return None

        image_url = response_data["artifacts"][0]["finishReason"]
        if not image_url:
            logging.error("No image URL found in Stability Diffusion response.")
            return None
        logging.info(f"Image URL: {image_url}")
        return response_data["artifacts"][0]["url"]

    except requests.exceptions.RequestException as e:
        logging.error(f"Error during Stability Diffusion API request: {e}")
        return None
    except json.JSONDecodeError:
        logging.error("Error decoding Stability Diffusion API response JSON.")
        return None
    except KeyError:
        logging.error("Error accessing 'artifacts' in Stability Diffusion API response.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during Stability Diffusion image generation: {e}", exc_info=True)
        return None

# --- Core Story Logic ---
async def get_poll_winner(bot: Bot, chat_id: str | int, message_id: int) -> Optional[str]:
    """Stops the specified poll and returns the winning option text, or None if no winner/error."""
    if message_id is None:
        logging.warning("No message ID provided to get_poll_winner.")
        return None

    logging.info(f"Attempting to stop poll (Message ID: {message_id})...")
    try:
        updated_poll: Poll = await bot.stop_poll(chat_id=chat_id, message_id=message_id)
        logging.info(f"Poll stopped (Message ID: {message_id}).")

        winning_options = []
        max_votes = -1
        for option in updated_poll.options:
            if option.voter_count > max_votes:
                max_votes = option.voter_count
                winning_options = [option.text]
            elif option.voter_count == max_votes and max_votes > 0:
                winning_options.append(option.text)

        if max_votes > 0 and len(winning_options) == 1:
            winner_text = winning_options[0]
            logging.info(f"Poll winner determined: '{winner_text}' ({max_votes} votes)")
            return winner_text
        elif max_votes > 0:
            winner_text = winning_options[0]
            logging.warning(f"Poll resulted in a tie ({len(winning_options)} options with {max_votes} votes). Picking first option: '{winner_text}'")
            return winner_text
        else:
            logging.info("Poll closed with no votes. Randomly selecting a winner.")
            if updated_poll.options:
                random_winner = random.choice(updated_poll.options)
                winner_text = random_winner.text
                logging.info(f"Randomly selected winner: '{winner_text}'")
                return winner_text
            else:
                logging.warning("Poll closed with no votes and no options found.")
                return None

    except telegram.error.BadRequest as e:
        err_text = str(e).lower()
        if "poll has already been closed" in err_text:
            logging.info(f"Poll (ID: {message_id}) was already closed. Attempting to fetch results directly (Currently not reliably implemented).", exc_info=True)
            return None
        elif "message to stop poll not found" in err_text:
            logging.error(f"Could not find the poll message to stop (ID: {message_id}). Was it deleted?")
            return None
        else:
            logging.error(f"Error stopping poll (BadRequest - ID: {message_id}): {e}")
            return None
    except telegram.error.Forbidden as e:
        logging.error(f"Error stopping poll (Forbidden - ID: {message_id}): {e}. Bot lacks permissions?", exc_info=True)
        raise
    except telegram.error.TelegramError as e:
        logging.error(f"Error stopping poll (ID: {message_id}): {e}", exc_info=True)
        return None

# --- Main Async Function ---
async def run_story_step():
    """Performs one step: loads state, gets winner, generates next step, posts, saves state."""
    if not validate_config():
        logging.critical("Configuration errors found. Exiting.")
        return

    logging.info("--- Running Story Step --- ")
    state = load_state()
    current_story = state.get("current_story", "")
    last_poll_message_id = state.get("last_poll_message_id")

    logging.info("Initializing Telegram bot...")
    bot = Bot(token=BOT_TOKEN)

    next_prompt = None
    story_just_started = False
    new_poll_message_id = None
    image_url: Optional[str] = None # Явное указание типа

    try:
        # 1. Get Poll Winner
        if last_poll_message_id:
            logging.info(f"Checking results for previous poll (ID: {last_poll_message_id})")
            poll_winner = await get_poll_winner(bot, CHANNEL_ID, last_poll_message_id)
            if poll_winner:
                next_prompt = poll_winner
            else:
                logging.warning(f"No winner determined from the last poll (ID: {last_poll_message_id}). Using initial idea or fallback.")
                next_prompt = INITIAL_STORY_IDEA
        else:
            logging.info("No last poll ID found in state. Using INITIAL_STORY_IDEA.")
            next_prompt = INITIAL_STORY_IDEA

        # 2. Generate & Post Story Part
        new_story_part = None
        if not current_story:
            logging.info("No existing story found. Posting initial idea.")
            message_to_send = INITIAL_STORY_IDEA
            current_story = INITIAL_STORY_IDEA
            story_just_started = True
            logging.info(f"Sending initial story part to channel {CHANNEL_ID}...")
            try:
                await bot.send_message(chat_id=CHANNEL_ID, text=message_to_send)
                logging.info("Initial story part sent.")
            except telegram.error.TelegramError as e:
                logging.error(f"Failed to send initial story part: {e}", exc_info=True)
                raise
        else:
            story_just_started = False
            if not next_prompt:
                logging.error("No prompt available for continuation (should not happen!). Using fallback.")
                next_prompt = "Продолжай как считаешь нужным."

            logging.info(f"Generating story continuation based on: '{next_prompt}'")
            new_story_part = generate_story_continuation_openai(current_story, next_prompt)
            if new_story_part and new_story_part.strip():
                combined_context = current_story + new_story_part
                image_url = generate_image_stability_diffusion(combined_context, image_style="pixel art") # Передаем стиль
                try:
                    if image_url:
                        await bot.send_photo(
                            chat_id=CHANNEL_ID,
                            photo=image_url,
                            caption=new_story_part
                        )
                        logging.info("Изображение и текст отправлены.")
                    else:
                         await bot.send_message(chat_id=CHANNEL_ID, text=new_story_part)
                         logging.info("Текст отправлен без изображения.")
                    current_story += new_story_part
                except telegram.error.Teleg