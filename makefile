# Digite "make" para instalar o ambiente (venv em .venv + pip).
PYTHON = python3
VENV = .venv
PY = $(VENV)/bin/python
MAIN_SCRIPT = main.py
YAML_TEST = read_yaml_example.py
SETUP_SCRIPT = requirements.txt

# Comando padrão: se digitar só make, já funciona
all: install test
# Cria .venv se não existir e instala dependências com python -m pip
install:
	@echo "🛠️ Preparando o ambiente..."
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r $(SETUP_SCRIPT)
# Testa o arquivo YAML (usa o Python do venv; rode make install antes)
test:
	@echo "🔍 Verificando o arquivo config.yaml..."
	@test -x $(PY) || (echo "Execute: make install" >&2 && exit 1)
	$(PY) $(YAML_TEST)
# Testes unitários com pytest (measurements, conversao, ia.visao, tracking)
test-unit:
	@test -x $(PY) || (echo "Execute: make install" >&2 && exit 1)
	$(PY) -m pytest -q
# Comando para rodar o projeto
run:
	@echo "Iniciando o ArrobaDigital"
	@test -x $(PY) || (echo "Execute: make install" >&2 && exit 1)
	$(PY) $(MAIN_SCRIPT)
# Modo Pessoas (experimental): peso humano ao vivo pela webcam
run-person:
	@echo "Iniciando ArrobaDigital — modo Pessoas (experimental)"
	@test -x $(PY) || (echo "Execute: make install" >&2 && exit 1)
	$(PY) person_demo.py --source 0 --known-height-cm 175
# Comando para limpar arquivos temporários
clean:
	@echo "🧹 Limpando arquivos temporários..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	@echo "✨ Limpeza concluída!"

# Para rodar o projeto, basta usar: make run