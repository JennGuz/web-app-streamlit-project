from dotenv import load_dotenv
from sqlalchemy import create_engine
import os

load_dotenv()

print(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")  # Esto te ayudará a verificar si DATABASE_URL fue cargado correctamente

def db_connect():
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine