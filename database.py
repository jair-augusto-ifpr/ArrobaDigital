import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def conectar():
    return psycopg2.connect(DATABASE_URL)

def iniciar_banco():
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            id SERIAL PRIMARY KEY,
            weight REAL,
            area REAL,
            length REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def salvar_registro(peso, area, comprimento):
    try:
        conn = conectar()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO measurements (weight, area, length)
            VALUES (%s, %s, %s)
        """, (peso, area, comprimento))
        conn.commit()
        conn.close()
        print(f"Dados salvos com sucesso no Neon: {peso}kg, {area}m², {comprimento}m")
    except Exception as e:
        print(f"Erro ao salvar no banco em nuvem: {e}")