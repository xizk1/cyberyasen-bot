# -*- coding: utf-8 -*-
import asyncio
import logging
import sys
import io
import os
import time
import gc
import urllib.parse
from datetime import datetime

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.types import InputFile
from aiogram.utils import executor
import numpy as np

warnings.filterwarnings("ignore")

# токен из переменных окружения
bot_token = os.environ.get("BOT_TOKEN")

class config:
    def __init__(self):
        self.model = "OFA-Sys/small-stable-diffusion-v0"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.steps = 25
        self.size = 384
        self.final_size = 512
        self.guidance_scale = 7.5
        self.max_queue = 1
        self.translations = {
            "котик": "cat, fluffy, cute, animal, detailed fur",
            "кот": "cat, fluffy, cute, animal, detailed fur",
            "кошка": "cat, fluffy, cute, animal, detailed fur",
            "собака": "dog, cute, animal, detailed fur, pet",
            "пёс": "dog, cute, animal, detailed fur, pet",
            "домик": "house, cottage, building, architecture, detailed",
            "дом": "house, building, architecture, detailed",
            "девушка": "beautiful girl, woman, detailed face, portrait",
            "парень": "handsome man, guy, detailed face, portrait",
            "робот": "robot, mechanical, futuristic, sci-fi, detailed",
            "машина": "car, vehicle, detailed, realistic, automotive",
            "автомобиль": "car, vehicle, detailed, realistic, automotive",
            "пейзаж": "landscape, nature, scenery, beautiful view",
            "гора": "mountain, landscape, nature, scenic view",
            "море": "sea, ocean, water, waves, beach, nature",
            "космос": "space, galaxy, stars, nebula, universe, cosmic",
            "космонавт": "astronaut, space suit, space, cosmos, realistic",
            "дракон": "dragon, mythical creature, fantasy, detailed",
        }

settings = config()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=bot_token)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

model_pipe = None
generation_queue = asyncio.Queue()
current_generations = 0
users_in_process = set()
model_loading = False

class memory_manager:
    def clear(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

memory = memory_manager()

def translate_prompt(text):
    text_lower = text.lower()
    
    for ru_word, en_prompt in settings.translations.items():
        if ru_word in text_lower:
            return f"{en_prompt}, {text}"
    
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=ru&tl=en&dt=t&q={urllib.parse.quote(text)}"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            translated = response.json()[0][0][0]
            return f"{translated}, detailed, realistic, high quality, 4k"
    except:
        pass
    
    return f"{text}, detailed, realistic, high quality, 4k"

async def load_model():
    global model_pipe, model_loading
    
    if model_loading:
        return False
        
    model_loading = True
    
    try:
        logger.info("загрузка модели...")
        
        model_pipe = StableDiffusionPipeline.from_pretrained(
            settings.model,
            torch_dtype=torch.float32,
            safety_checker=None,
            low_cpu_mem_usage=True
        )
        
        model_pipe = model_pipe.to(settings.device)
        
        model_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            model_pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            final_sigmas_type="sigma_min"
        )
        
        if settings.device == "cpu":
            model_pipe.enable_attention_slicing()
        
        logger.info(f"модель загружена на {settings.device}")
        return True
    except Exception as e:
        logger.error(f"ошибка загрузки: {e}")
        return False
    finally:
        model_loading = False

async def generate_image(prompt: str, user_id: int):
    global current_generations
    
    try:
        memory.clear()
        
        translated = translate_prompt(prompt)
        logger.info(f"user {user_id}: {prompt} -> {translated}")
        
        start_time = time.time()
        
        with torch.no_grad():
            image = model_pipe(
                translated,
                negative_prompt="bad quality, blurry, distorted, ugly, low resolution",
                num_inference_steps=settings.steps,
                height=settings.size,
                width=settings.size,
                guidance_scale=settings.guidance_scale,
            ).images[0]
        
        image = image.resize((settings.final_size, settings.final_size))
        
        elapsed = time.time() - start_time
        
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='png')
        img_bytes.seek(0)
        
        return img_bytes, elapsed
        
    except Exception as e:
        logger.error(f"ошибка генерации: {e}")
        raise
    finally:
        current_generations -= 1
        if user_id in users_in_process:
            users_in_process.remove(user_id)
        memory.clear()

async def process_queue():
    while True:
        if current_generations < settings.max_queue and not generation_queue.empty():
            user_id, prompt, message = await generation_queue.get()
            
            try:
                await message.edit_text("генерация началась...")
                
                img_bytes, elapsed = await generate_image(prompt, user_id)
                
                caption = f"готово за {elapsed:.1f} сек\nзапрос: {prompt}"
                
                await message.delete()
                await bot.send_photo(
                    user_id,
                    InputFile(img_bytes, filename=f"img_{int(time.time())}.png"),
                    caption=caption
                )
                
            except Exception as e:
                logger.error(f"ошибка: {e}")
                try:
                    await bot.send_message(user_id, f"ошибка: {str(e)[:100]}")
                except:
                    pass
                
        await asyncio.sleep(1)

@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    welcome_text = """
cyberyasen bot

пришли текст - получи картинку

примеры:
- котик в космосе
- закат на море
- робот киберпанк

/help - помощь
/stats - статистика
    """
    await message.reply(welcome_text)

@dp.message_handler(commands=['help'])
async def cmd_help(message: types.Message):
    help_text = """
советы:
- пиши конкретно
- используй прилагательные
- жди 30-60 секунд
    """
    await message.reply(help_text)

@dp.message_handler(commands=['stats'])
async def cmd_stats(message: types.Message):
    stats_text = f"""
очередь: {current_generations}/{settings.max_queue}
в ожидании: {generation_queue.qsize()}
устройство: {settings.device}
    """
    await message.reply(stats_text)

@dp.message_handler()
async def handle_prompt(message: types.Message):
    global current_generations
    
    user_id = message.from_user.id
    prompt = message.text.strip()
    
    if model_pipe is None:
        if model_loading:
            await message.reply("модель загружается... подожди")
        else:
            await message.reply("модель не загружена. пиши /start")
        return
    
    if user_id in users_in_process:
        await message.reply("уже генерирую, подожди")
        return
    
    if len(prompt) < 3:
        await message.reply("короткий запрос")
        return
    
    status_msg = await message.reply(
        f"запрос принят\n"
        f"промпт: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n"
        f"позиция: {generation_queue.qsize() + 1}"
    )
    
    users_in_process.add(user_id)
    await generation_queue.put((user_id, prompt, status_msg))

@dp.message_handler(content_types=['photo', 'document', 'voice', 'video'])
async def handle_unsupported(message: types.Message):
    await message.reply("только текст")

async def on_startup(dp):
    asyncio.create_task(load_model())
    asyncio.create_task(process_queue())
    logger.info("бот запущен")

async def on_shutdown(dp):
    await bot.close()
    memory.clear()
    logger.info("бот остановлен")

if __name__ == "__main__":
    if not bot_token:
        print("нет токена бота")
        sys.exit(1)
    
    executor.start_polling(
        dp, 
        on_startup=on_startup, 
        on_shutdown=on_shutdown,
        skip_updates=True,
        timeout=30
    )
