import os
from pathlib import Path

import psycopg2
from dotenv import dotenv_values, load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _resolve_database_url():
    """Lê DATABASE_URL do .env (arquivo) e do ambiente; dotenv_values evita falhas de parse do load_dotenv."""
    vals = dotenv_values(_ENV_PATH) or {}
    for key in ("DATABASE_URL", "POSTGRES_URL"):
        raw = vals.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip().strip('"').strip("'")
    load_dotenv(_ENV_PATH, override=True)
    for key in ("DATABASE_URL", "POSTGRES_URL"):
        raw = os.getenv(key)
        if raw and str(raw).strip():
            return str(raw).strip().strip('"').strip("'")
    return None


DATABASE_URL = _resolve_database_url()
DB_DISPONIVEL = True

def conectar():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL não definida no ambiente")
    return psycopg2.connect(DATABASE_URL)

def iniciar_banco():
    global DB_DISPONIVEL
    if not DATABASE_URL:
        DB_DISPONIVEL = False
        print(
            "[AVISO] Sem DATABASE_URL no .env — seguindo sem gravar no Postgres. "
            "Crie/edite .env na raiz com: DATABASE_URL=postgresql://usuario:senha@host:5432/banco"
        )
        return
    try:
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
        DB_DISPONIVEL = True
    except Exception as e:
        DB_DISPONIVEL = False
        print(f"[AVISO] Banco indisponível. Execução seguirá sem persistência: {e}")

def salvar_registro(peso_kg, area_m2, altura_m, largura_m, area_cm2, altura_cm, largura_cm):
    if not DB_DISPONIVEL:
        return
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


def ultimos_registros(limit: int = 20):
    """Retorna os últimos `limit` registros da tabela `measurements` como lista de dicts.

    Ordenados do mais recente para o mais antigo. Retorna lista vazia se o banco não estiver disponível.
    """
    if not DB_DISPONIVEL or not DATABASE_URL:
        return []
    try:
        conn = conectar()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, weight_kg, area_m2, height_m, width_m, area_cm2, height_cm, width_cm, timestamp
            FROM measurements
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (int(limit),),
        )
        rows = cursor.fetchall()
        conn.close()
        registros = []
        for r in rows:
            registros.append({
                "id": r[0],
                "peso_kg": r[1],
                "area_m2": r[2],
                "altura_m": r[3],
                "largura_m": r[4],
                "area_cm2": r[5],
                "altura_cm": r[6],
                "largura_cm": r[7],
                "timestamp": r[8],
            })
        return registros
    except Exception as e:
        print(f"[AVISO] Falha ao ler registros do banco: {e}")
        return []