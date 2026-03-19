from database import iniciar_banco, salvar_registro

def processar_dados(peso, area, comprimento):
    try:
        salvar_registro(peso, area, comprimento)
        return True
    except Exception as e:
        print(f"Erro no Armazenamento: {e}")
        return False

if __name__ == "__main__":
    iniciar_banco()

    peso_teste = 420.0
    area_teste = 0.95
    comp_teste = 1.75

    print("Iniciando Armazenamento no Neon...")

    if processar_dados(peso_teste, area_teste, comp_teste):
        print(f"Dados salvos {peso_teste}kg, {area_teste}m², {comp_teste}m")