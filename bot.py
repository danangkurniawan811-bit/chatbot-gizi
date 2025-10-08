import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
import os
from dotenv import load_dotenv
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Selamat datang! Gunakan /askgizi <pertanyaan> untuk bertanya tentang gizi.")

async def answer(update: Update, context: CallbackContext):
    query = " ".join(context.args)
    flask_url = os.getenv("FLASK_API_URL")

    if not flask_url:
        await update.message.reply_text("Server belum terhubung. Coba lagi nanti ya 🙏")
        return

    try:
        response = requests.post(f"{flask_url}/ask", json={"query": query}).json()
        await update.message.reply_text(
            response.get("answer", "Maaf, saya tidak dapat menemukan jawaban untuk pertanyaan Anda.")
        )
    except Exception as e:
        await update.message.reply_text(f"Terjadi kesalahan: {e}")

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("askgizi", answer))
    application.run_polling()

if __name__ == "__main__":
    main()
