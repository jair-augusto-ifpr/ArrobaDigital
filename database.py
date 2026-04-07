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
            weight_kg REAL,
            area_m2 REAL,
            height_m REAL,
            width_m REAL,
            area_cm2 REAL,
            height_cm REAL,
            width_cm REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS weight_kg REAL")
    cursor.execute("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS area_m2 REAL")
    cursor.execute("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS height_m REAL")
    cursor.execute("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS width_m REAL")
    cursor.execute("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS area_cm2 REAL")
    cursor.execute("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS height_cm REAL")
    cursor.execute("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS width_cm REAL")
    conn.commit()
    conn.close()

def salvar_registro(peso_kg, area_m2, altura_m, largura_m, area_cm2, altura_cm, largura_cm):
    try:
        conn = conectar()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO measurements (
                weight_kg,
                area_m2,
                height_m,
                width_m,
                area_cm2,
                height_cm,
                width_cm
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (peso_kg, area_m2, altura_m, largura_m, area_cm2, altura_cm, largura_cm))
        conn.commit()
        conn.close()
        print(
            "Dados salvos com sucesso no banco: "
            f"{peso_kg:.2f}kg, {area_m2:.4f}m², {altura_m:.2f}m x {largura_m:.2f}m"
        )
    except Exception as e:
        print(f"Erro ao salvar no banco em nuvem: {e}")