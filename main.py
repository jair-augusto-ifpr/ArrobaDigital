from database import iniciar_banco, salvar_registro

def processar_dados(peso, area, comprimento):
    try:
        iniciar_banco()
        salvar_registro()
        
        return True
    except Exception as e: 
        print(f"Erro no Armazenamento: {e}"
        return False
        
        
if __name__ == "__main__"
    #Dados passados na task
    peso_teste = 420.0
    area_teste = 0.95
    comp_teste = 1.75
    
    print("Iniciando Armazenamento")
    
    if processar_dados(peso_teste, area_teste, comp_teste):
        print("Armazenamento Finalizado com sucesso")