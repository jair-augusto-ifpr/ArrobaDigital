# Digite "make" ara instalar o ambiente
PYTHON = python
PIP = pip
SETUP_SCRIPT = setup_project.py
YAML_TEST = read_yaml_example.py
MAIN_SCRIPT = main.py

# Comando padrão: se digitar só make, ja funciona
all: install test
# So intala o ambiente
install:
	@echo "🛠️ Preparando o ambiente..."
	$(PYTHON) $(SETUP_SCRIPT)
# So testa o arquivo YAML
test:
	@echo "🔍 Verificando o arquivo config.yaml..."
	$(PYTHON) $(YAML_TEST)
# Comando para rodar o projeto
run:
	@echo "🚀 Iniciando o BioWeight_PR no Paraná..."
	$(PYTHON) $(MAIN_SCRIPT)
# Comando para limpar arquivos temporários
clean:
	@echo "🧹 Limpando arquivos temporários..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	@echo "✨ Limpeza concluída!"

# Para rodar o projeto, basta usar: make run