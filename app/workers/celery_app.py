import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

celery = Celery(
    "codeqa",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    include=["app.workers.tasks"],
)
