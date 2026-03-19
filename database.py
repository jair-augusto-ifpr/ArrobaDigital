import sqlite3

def iniciar_banco():
    conn = sqlite3.connect("cattle.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        weight REAL,
        area REAL,
        length REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()
    
def salvar_registro(peso, area, comprimento):
    conn = sqlite3.connect("cattle.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO measurements (weight, area, length)
        VALUES (?, ?, ?)
    """, (peso, area, comprimento))
    conn.commit()
    conn.close()
    print(f"Dados salvos {peso}kg, {area}m², {comprimento}m")