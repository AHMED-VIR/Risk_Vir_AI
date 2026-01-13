import logging
import os
from io import BytesIO
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from google import genai
from google.genai import types

# --- Configuration ---
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
# Get tokens from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REQUIRED_CHANNEL = os.getenv("REQUIRED_CHANNEL", "@RISK_VIR")

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
    raise ValueError("Missing API keys. Please set TELEGRAM_BOT_TOKEN and GEMINI_API_KEY in .env file.")

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- Gemini Client Setup (New SDK) ---
client = genai.Client(api_key=GEMINI_API_KEY)

async def check_subscription(user_id: int, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Checks if the user is a member of the required channel.
    """
    try:
        member = await context.bot.get_chat_member(chat_id=REQUIRED_CHANNEL, user_id=user_id)
        if member.status in ['member', 'administrator', 'creator', 'restricted']:
            return True
        return False
    except Exception as e:
        logging.error(f"Error checking subscription for user {user_id}: {e}")
        # If bot is not admin or channel is invalid, fail closed (restrict access) or open (allow).
        # Returning False restricts access.
        return False

async def get_gemini_response(user_query: str, image_data: BytesIO = None) -> str:
    """
    Sends the user's query (and optional image) to Gemini API and returns the response.
    Tries multiple model names to handle API variations.
    """
    # List of models to try in order of preference
    # List of models to try in order of preference
    # Prioritizing Flash models for speed and higher rate limits on free tier
    candidate_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash-001",
        "gemini-flash-latest",
        "gemini-pro-latest",
        "gemini-1.5-flash",
        "gemini-1.5-flash-001",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-8b",
        "gemini-2.0-flash-exp",
    ]

    last_error = None

    for model_name in candidate_models:
        try:
            if image_data:
                # Open image with PIL (reset buffer position if needed, though usually fine)
                image_data.seek(0)
                image = Image.open(image_data)
                
                # If query is empty, provide a default prompt
                prompt_text = user_query if user_query else "Describe this image."
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt_text, image]
                )
            else:
                if not user_query:
                    return "Please send a message."

                response = client.models.generate_content(
                    model=model_name,
                    contents=user_query
                )
            
            # If successful, return immediately
            return response.text

        except Exception as e:
            logging.warning(f"Failed with model {model_name}: {e}")
            last_error = e
            continue
    
    # If all failed
    logging.error(f"All Gemini models failed. Last error: {last_error}")
    return f"عذراً، حدث خطأ أثناء الاتصال بـ Gemini API.\nفشلت جميع المحاولات.\nالخطأ الأخير: {last_error}"


async def get_image_generation(user_prompt: str) -> BytesIO:
    """
    Generates an image using Imagen model.
    """
    try:
        # Use a model from the list provided by user, start with the fast one
        model_name = "imagen-4.0-fast-generate-001"
        
        response = client.models.generate_images(
            model=model_name,
            prompt=user_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
            )
        )
        
        # Access the first generated image
        if response.generated_images:
            image_bytes = response.generated_images[0].image.image_bytes
            return BytesIO(image_bytes)
        return None
    except Exception as e:
        logging.error(f"Error generating image: {e}")
        return None

# --- Telegram Bot Handlers ---

async def generate_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for /image command.
    """
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    is_member = await check_subscription(user_id, context)
    if not is_member:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"⚠️ **عذراً، يجب عليك الاشتراك في القناة أولاً:**\n{REQUIRED_CHANNEL}"
        )
        return

    # Extract prompt
    if not context.args:
        await context.bot.send_message(
            chat_id=chat_id,
            text="يرجى كتابة وصف للصورة بعد الأمر، مثال:\n`/image قطة تركب دراجة`",
            parse_mode='Markdown'
        )
        return

    user_prompt = " ".join(context.args)
    
    await context.bot.send_chat_action(chat_id=chat_id, action='upload_photo')
    
    try:
        image_data = await get_image_generation(user_prompt)
        
        if image_data:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=image_data,
                caption=f"🎨 **Generated Image**\nPromp: {user_prompt}"
            )
        else:
             await context.bot.send_message(
                chat_id=chat_id,
                text="عذراً، لم أتمكن من إنشاء الصورة. يرجى المحاولة مرة أخرى لاحقاً."
            )
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"حدث خطأ: {e}"
        )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for the /start command.
    """
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    is_member = await check_subscription(user_id, context)
    if not is_member:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"⚠️ **عذراً، لا يمكنك استخدام البوت إلا بعد الاشتراك في القناة.**\n\n"
                f"يرجى الاشتراك هنا: {REQUIRED_CHANNEL}\n\n"
                "بعد الاشتراك، اضغط على /start مرة أخرى للبدء."
            )
        )
        return

    await context.bot.send_message(
        chat_id=chat_id,
        text=(
            "مرحباً! أنا بوت ذكي يعمل بواسطة Google Gemini AI. 🤖✨\n\n"
            "💬 **للمحادثة:** أرسل أي نص أو سؤال.\n"
            "📸 **لتحليل الصور:** أرسل صورة وسأصفها لك.\n"
            "🎨 **لإنشاء الصور:** استخدم الأمر `/image` متبوعاً بالوصف.\n"
            "   مثال: `/image منظر طبيعي للجبال وقت الغروب`"
        )
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for text messages and photos.
    """
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Check subscription
    is_member = await check_subscription(user_id, context)
    if not is_member:
         await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"⚠️ **عذراً، يجب عليك الاشتراك في القناة أولاً.**\n\n"
                f"القناة: {REQUIRED_CHANNEL}\n"
            )
        )
         return

    # Notify user that we are processing
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    user_text = ""
    img_buffer = None

    # Handle Photo
    if update.message.photo:
        # Get the largest photo (last in the list)
        photo_file = await update.message.photo[-1].get_file()
        
        # Download into memory
        img_buffer = BytesIO()
        await photo_file.download_to_memory(out=img_buffer)
        img_buffer.seek(0)
        
        # Use caption as text if available
        user_text = update.message.caption
    
    # Handle Text
    elif update.message.text:
        user_text = update.message.text
    
    else:
        return

    # Get response from Gemini
    response_text = await get_gemini_response(user_text, img_buffer)

    # Send the response back to the user
    await context.bot.send_message(
        chat_id=chat_id,
        text=response_text
    )

# --- Main Execution ---

if __name__ == '__main__':
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("Error: Please replace 'YOUR_TELEGRAM_BOT_TOKEN_HERE' with your actual Telegram bot token in bot.py")
    else:
        # Build the application
        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

        # Add handlers
        start_handler = CommandHandler('start', start)
        image_handler = CommandHandler('image', generate_image_command)
        message_handler = MessageHandler((filters.TEXT | filters.PHOTO) & (~filters.COMMAND), handle_message)

        application.add_handler(start_handler)
        application.add_handler(image_handler)
        application.add_handler(message_handler)

        print("Bot is polling...")
        # Run the bot
        application.run_polling()
