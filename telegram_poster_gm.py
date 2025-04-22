import asyncio
import telegram
from telegram import Bot, Poll, Message
import json
import os
from pathlib import Path
import logging
import random
import requests
import google.generativeai as genai  # Для Gemini # ADDED google.generativeai

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename="thecode.log")

# --- Configuration ---
# !! IMPORTANT: Замените placeholders ниже на ваши актуальные значения !!
BOT_TOKEN = "8058100768:AAFpSm1gmmKdZallAnFG1UogRHaByY7JYe4"  # <-- Вставьте ваш BOT TOKEN здесь
CHANNEL_ID = "@testovidhe"  # Или "-100xxxxxxxxxx" для приватных каналов/групп
#BOT_TOKEN = os.getenv("BOT_TOKEN", "8058100768:AAFpSm1gmmKdZallAnFG1UogRHaByY7JYe4")  # Замените на ваш ключ Gemini API
#CHANNEL_ID = os.getenv("CHANNEL_ID", "@testovidh")  # Замените на ваш ключ Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAeF6jq3CzxdvXpUxW9YpgInSE6ov51Blo")  # Замените на ваш ключ Gemini API

# Story settings
INITIAL_STORY_IDEA = """
Альтернативная вселенная: Великая Отечественная война + киберпанк 
Главный герой: Солдат красной армии.

Солдата зовут Андрей, он и его отряд на пути к Берлину.

Берлинский маршрут: Протокол 77-Б
Ночью на них напал дождь.
Не обычный — асинхронный. Капли падали до того, как небо начинало хмуриться.
"Погодная инверсия", — сказал Клим, просканировав фронт.
— Немцы перехватили цикл времени, — добавил он. — Опять играются с фазами.

Молчанов ткнул пальцем в небо.
Там — силуэт дирижабля. Только у него не было корпуса. Один только каркас — светящийся, как нервная система.

Нейроаэростат. Платформа вещания.
С него транслировали речь Гитлера, но… искажённую. Словно голос прошёл через тысячу фильтров и слипся в сплошной шёпот.
Он бил прямо в кость.

— Не слушайте, — сказал Андрей.

В голове зашевелилось.
“Отступай”, — говорил внутренний голос.
“Ты не победишь”.
“Ты — не тот, кем был”.
Андрей сжал кулак. На ладони — символ: маленькая пятиконечная звезда. Механическая. Пульсирующая.

Это был знак “функции возврата” — последняя возможность выйти из цикла.

"""  # The very first story prompt (in Russian)

# ALZ0: Добавлены настройки генерации изображений
IMAGE_MODEL = "gemini-2.0-flash"  #  Gemini Pro Vision для генерации изображений
IMAGE_STYLE = "pixel art"  # Стиль пиксельной графики
IMAGE_SIZE = "1024x1024"  # Размер изображения (поддерживается Gemini)

STATE_FILE = Path(__file__).parent / "story_state.json"  # File to store story progress
POLL_QUESTION_TEMPLATE = "Как продолжится история?"  # Default question for polls

# Gemini Settings
GEMINI_MODEL = "gemini-2.0-flash"  #  Gemini Pro для текста
MAX_CONTEXT_CHARS = 15000  # Approximate limit to avoid huge API requests (adjust as needed)

# --- End Configuration ---

# --- Gemini Client Initialization ---
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model_text = genai.GenerativeModel(GEMINI_MODEL)
gemini_model_image = genai.GenerativeModel(IMAGE_MODEL)

if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY":
    logging.info("Gemini client initialized successfully.")
else:
    logging.warning("GOOGLE_API_KEY not found or is placeholder. LLM features will be disabled.")


# ALZ0: Новая функция для генерации пиксель-арта
async def generate_pixel_art(story_context: str) -> str | None:
    """Генерирует URL изображения в стиле пиксель-арт через Gemini"""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        logging.warning("Gemini client not available. Skipping image generation.")
        return None

    try:
        prompt = f"Generate pixel art in the style of {IMAGE_STYLE} for the scene from the story: {story_context[:800]}"
        response = gemini_model_image.generate_content(prompt)

        if response.candidates and response.candidates[0].content.parts:
            image_data = response.candidates[0].content.parts[0].inline_data.data
            #  Здесь нужно сохранить image_data как файл и вернуть его путь/URL
            #  Пример (требует доработки):
            image_path = "temp_image.png"  #  Временный путь
            with open(image_path, "wb") as f:
                f.write(image_data)
            return image_path  #  Или возвращать URL, если вы загружаете куда-то
        else:
            logging.error("No image data in Gemini response.")
            return None

    except Exception as e:
        logging.error(f"Ошибка генерации изображения: {e}")
        return None


# --- State Management ---
def load_state():
    """Loads the story state (current_story, last_poll_message_id) from the JSON file."""
    default_state = {"current_story": "", "last_poll_message_id": None}
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
                if "current_story" not in state:
                    state["current_story"] = default_state["current_story"]
                if "last_poll_message_id" not in state:
                    state["last_poll_message_id"] = default_state["last_poll_message_id"]
                logging.info(f"State loaded from {STATE_FILE}: {state}")
                return state
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading state file {STATE_FILE}: {e}. Starting fresh.")
            return default_state
    else:
        logging.info("State file not found. Starting fresh.")
        return default_state


def save_state(state):
    """Saves the story state to the JSON file."""
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
        logging.info(f"Story state saved to {STATE_FILE}: {state}")
    except IOError as e:
        logging.error(f"Error saving state file {STATE_FILE}: {e}")


# --- Helper Function to Validate Configuration ---
def validate_config():
    """Checks if the configuration values have been changed from placeholders."""
    valid = True
    if BOT_TOKEN == "YOUR_BOT_TOKEN" or not BOT_TOKEN:
        logging.error("BOT_TOKEN is not set correctly.")
        valid = False
    if not CHANNEL_ID or CHANNEL_ID == "YOUR_CHANNEL_ID":
        logging.error("CHANNEL_ID is not set correctly.")
        valid = False
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        logging.warning("GOOGLE_API_KEY is not set correctly. LLM features disabled.")
        valid = False
    if not INITIAL_STORY_IDEA:
        logging.error("INITIAL_STORY_IDEA cannot be empty.")
        valid = False
    return valid


# --- Gemini Interaction Functions ---
async def generate_story_continuation_gemini(current_story: str, user_choice: str) -> str | None:
    """Calls the Gemini API to get the next story part.

    Returns:
        The new story part string, or None if API call fails.
    """
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        logging.warning("Gemini client not available. Skipping story generation.")
        return "\n\n[Продолжение не сгенерировано - Gemini недоступен]"

    logging.info("Generating story continuation via Gemini...")

    truncated_story = current_story
    if len(current_story) > MAX_CONTEXT_CHARS:
        logging.warning(
            f"Current story context ({len(current_story)} chars) exceeds limit ({MAX_CONTEXT_CHARS}). Truncating.")
        truncated_story = current_story[-MAX_CONTEXT_CHARS:]

    prompt = f"""Ты - самый великий современный творческий писатель, продолжающий интерактивную историю на русском языке. 
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

Предыдущая история:
{truncated_story}

Выбор пользователя: '{user_choice}'

Напиши следующие три параграфа, используя инструмент 'write_story_part'."""

    try:
        response = gemini_model_text.generate_content(prompt)
        if response.text:
            try:
                arguments = json.loads(response.text)
                new_story_part = arguments.get("story_part")
                reasoning = arguments.get("reasoning")
                if new_story_part:
                    logging.info(f"Successfully parsed story part: {new_story_part[:100]}...")
                    return new_story_part
                else:
                    logging.error("Gemini response missing 'story_part'")
                    return None
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse Gemini response: {e}")
                return None
        else:
            logging.error("Gemini returned an empty response.")
            return None

    except Exception as e:
        logging.error(f"Gemini API error during story generation: {e}", exc_info=True)
        return None


async def generate_poll_options_gemini(full_story_context: str) -> list[str] | None:
    """Calls the Gemini API to get 4 poll options.

    Returns:
        A list of 4 distinct poll options (max 90 chars each), or None if API call fails.
    """
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        logging.warning("Gemini client not available. Skipping poll option generation.")
        return None

    logging.info("Generating poll options via Gemini...")

    truncated_context = full_story_context[-MAX_CONTEXT_CHARS:]

    prompt = f"""Ты - помощник для интерактивной истории на русском языке. 
Тебе дан ПОЛНЫЙ текущий текст истории. Твоя задача - придумать ровно 4 КОРОТКИХ (максимум 90 символов!) и ФУНДАМЕНТАЛЬНО РАЗНЫХ варианта продолжения сюжета для опроса в Telegram. 
Варианты должны быть МАКСИМАЛЬНО НЕПОХОЖИМИ друг на друга, предлагая совершенно разные, возможно, даже противоположные, направления развития событий (например, пойти на север ИЛИ пойти на юг ИЛИ остаться на месте ИЛИ искать что-то конкретное).
Избегай незначительных вариаций одного и того же действия. Нужны действительно ОТЛИЧАЮЩИЕСЯ выборы.
Возвращай результат ТОЛЬКО в формате JSON, используя предоставленный инструмент 'suggest_poll_options' с полем 'options' (массив из 4 строк). Не добавляй никакого другого текста.

Полный текст текущей истории:
{truncated_context}

Предложи 4 варианта для опроса, используя инструмент 'suggest_poll_options'."""

    try:
        response = gemini_model_text.generate_content(prompt)
        if response.text:
            try:
                args = json.loads(response.text)
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
        else:
            logging.error("Gemini returned an empty response.")
            return None

    except Exception as e:
        logging.error(f"Poll generation failed: {str(e)}")
        return None


# --- Core Story Logic ---
async def get_poll_winner(bot: Bot, chat_id: str | int, message_id: int) -> str | None:
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
            logging.warning(
                f"Poll resulted in a tie ({len(winning_options)} options with {max_votes} votes). Picking first option: '{winner_text}'")
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
            logging.info(f"Poll (ID: {message_id}) was already closed. Attempting to fetch results directly (Currently not reliably implemented).",
                         exc_info=True)
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

    try:
        # 1. Get Poll Winner (if applicable)
        if last_poll_message_id:
            logging.info(f"Checking results for previous poll (ID: {last_poll_message_id})")
            poll_winner = await get_poll_winner(bot, CHANNEL_ID, last_poll_message_id)
            if poll_winner:
                next_prompt = poll_winner
            else:
                logging.warning(
                    f"No winner determined from the last poll (ID: {last_poll_message_id}). Using initial idea or fallback.")
                next_prompt = INITIAL_STORY_IDEA
        else:
            logging.info("No last poll ID found in state. Using INITIAL_STORY_IDEA.")
            next_prompt = INITIAL_STORY_IDEA

        # 2. Generate & Post Story Part
        new_story_part = None
        image_url = None
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
            new_story_part = await generate_story_continuation_gemini(current_story, next_prompt)
            if new_story_part and new_story_part.strip():
                combined_context = current_story + new_story_part
                image_url = await generate_pixel_art(combined_context)
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
                except telegram.error.TelegramError as e:
                    logging.error(f"Failed to send new story part: {e}", exc_info=True)
                    raise
            else:
                logging.error("Story continuation failed or returned empty. Story not updated. Interrupting step.")
                raise RuntimeError("LLM failed to generate story continuation.")

        # 3. Generate and Post Poll
        logging.info("Generating poll options based on current story...")
        poll_options = await generate_poll_options_gemini(current_story) or [
            "Продолжить штурмовать позиции",
            "Искать обходной путь",
            "Запросить подкрепление",
            "Перегруппироваться"
        ]

        if not poll_options:
            logging.warning("Using fallback poll options")
            poll_options = ["Атаковать с фланга",
                               "Укрепить оборону",
                               "Провести разведку",
                               "Изменить стратегию"]

        if not poll_options or len(poll_options) != 4:
            logging.error("Could not generate valid poll options. Skipping poll posting.")
            new_poll_message_id = None
        else:
            truncated_options = [opt[:90] for opt in poll_options]
            logging.info(
                f"Generated {len(truncated_options)} poll options (truncated if needed). First option: '{truncated_options[0]}'...")
            try:
                sent_poll_message: Message = await bot.send_poll(
                    chat_id=CHANNEL_ID,
                    question=POLL_QUESTION_TEMPLATE,
                    options=truncated_options,
                    is_anonymous=True,
                )
                new_poll_message_id = sent_poll_message.message_id
                logging.info(f"New poll sent (Message ID: {new_poll_message_id}).")
            except telegram.error.TelegramError as poll_error:
                logging.error(f"Error sending poll: {poll_error}. Skipping poll posting.", exc_info=True)
                new_poll_message_id = None

        # 4. Save State for Next Run
        state_to_save = {
            "current_story": current_story,
            "last_poll_message_id": new_poll_message_id
        }
        save_state(state_to_save)
        logging.info("--- Story Step Completed Successfully --- ")

    except google.generativeai.APIError as e:
        logging.error(f"\n--- A Gemini API Error Occurred During Story Step --- ")
        logging.error(f"Error message: {e}")
        logging.error("Script interrupted due to Gemini API error. State NOT saved for this run.")
    except telegram.error.TelegramError as e:
        logging.error(f"\n--- A Telegram API Error Occurred During Story Step --- ")
        logging.error(f"Error message: {e}")
        logging.error("Script interrupted due to Telegram API error. State NOT saved for this run.")
    except RuntimeError as e:
        logging.error(f"\n--- A Runtime Error Occurred During Story Step --- ")
        logging.error(f"Error message: {e}")
        logging.error("Script interrupted. State NOT saved for this run.")
    except Exception as e:
        logging.error(f"\n--- An Unexpected Error Occurred During Story Step --- ")
        logging.error(f"Error message: {e}", exc_info=True)
        logging.error("Script interrupted due to unexpected error. State NOT saved for this run.")
    finally:
        logging.info("--- Story Step Finished --- ")


# --- Run the Script ---
if __name__ == "__main__":
    logging.info("Script execution started.")
    print(f"BOT_TOKEN value: {BOT_TOKEN}")  # Add this line
    print(f"CHANNEL_ID value: {CHANNEL_ID}")  # Add this line
    
    if not validate_config():
        logging.critical(
            "Configuration validation failed. Please check BOT_TOKEN, CHANNEL_ID, and GOOGLE_API_KEY (if used). Exiting.")
    elif not GOOGLE_API_KEY:
        logging.warning("Gemini clientnot initialized. LLM features will use placeholders or fail. Proceeding...")
        asyncio.run(run_story_step())
    else:
        logging.info("Configuration validated. Running async story step.")
        asyncio.run(run_story_step())

    logging.info("Script execution finished.")
