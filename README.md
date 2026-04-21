# ArrobaDigital

ArrobaDigital usa visão computacional (YOLOv8-seg) para detectar bovinos em vídeo/imagem, estimar medidas corporais a partir da máscara de segmentação e calcular o peso aproximado por regressão. Saída na tela (OpenCV) e, opcionalmente, gravação no Postgres.

## Estrutura do projeto

| Caminho | O que faz |
| --- | --- |
| `main.py` | Orquestra o pipeline: lê CLI/config, inicia câmera, chama IA, desenha HUD e salva no banco. |
| `config.yaml` | Parâmetros do projeto: escala da câmera, filtros de detecção, raça-foco. |
| `database.py` | Conecta ao Postgres via `DATABASE_URL` (arquivo `.env`). Se ausente, segue sem gravar. |
| `src/camera/capture.py` | Captura de vídeo em thread (webcam, arquivo, URL). |
| `src/detection/yolo_detector.py` | Carrega YOLOv8-seg e expõe `detect`, `track` (ByteTrack), `detect_all`, `filter_cows`. |
| `src/segmentation/mask_segmenter.py` | Converte as máscaras do YOLO em crops binários. |
| `src/biometrics/measurements.py` | Calcula área, comprimento e largura em cm a partir da máscara. |
| `src/conversao/conversao.py` | `modelo_regressao`, `modelo_biometrico` (PT²) e `estimar_peso` (dispatcher). |
| `src/tracking/aggregator.py` | EMA por `track_id` + política de salvamento (1x/boi, cooldown). |
| `src/utils/image_utils.py` | Funções gráficas: `improve_lighting`, `draw_boxes`, `draw_hud`, `draw_ia_panel`. |
| `src/ia/client.py` | Cliente OpenRouter (texto + visão, retry com backoff para 429/5xx). |
| `src/ia/laudo.py` | Laudo textual por boi a partir das medidas + peso. |
| `src/ia/visao.py` | Análise visual: raça provável, ECC (1–5), pelagem, observações. |
| `src/ia/relatorio.py` | Resumo do lote em linguagem natural via `--report`. |
| `src/logger/logger.py` | Logger padronizado. |
| `src/person/weight.py` | Modo Pessoas (experimental) — peso humano a partir da máscara. |
| `person_demo.py` | Script standalone do modo Pessoas (câmera + peso ao vivo). |
| `tests/` | Suite pytest (measurements, conversao, ia.visao, tracking.aggregator, person). |
| `read_yaml_example.py` | Valida `config.yaml` (usado por `make test`). |

## Quickstart

```bash
make install         # cria .venv e instala requirements.txt
make test            # valida o config.yaml
make test-unit       # roda a suite pytest (tests/)
make run             # roda com webcam (src=0), usando config.yaml
```

Para rodar com outros parâmetros, use `python` do venv direto:

```bash
# vídeo local, desenhando todas as classes que o YOLO enxerga
.venv/bin/python main.py --source caminho/do/video.mp4 --show-all

# imagem estática (jpg/png/...) — mostra o resultado e espera tecla
.venv/bin/python main.py --source caminho/boi.jpg

# sem banco, salvando saída anotada em mp4
.venv/bin/python main.py --source 0 --no-db --save-video saida.mp4

# ajustar confiança mínima do modelo
.venv/bin/python main.py --source 0 --conf 0.15 --show-all
```

### CLI

| Flag | Descrição |
| --- | --- |
| `--source`, `-s` | `0/1/...` (webcam), caminho de vídeo/imagem ou URL. Padrão `0`. |
| `--conf` | Confiança mínima do YOLO (sobrescreve `config.yaml processing.conf`). |
| `--show-all` | Desenha todas as classes detectadas (bois em verde, outros em amarelo). |
| `--save-video PATH` | Salva o vídeo anotado (mp4). |
| `--no-db` | Ignora o Postgres (nem tenta conectar). |
| `--no-ia` | Desativa chamadas à LLM (OpenRouter). |
| `--no-track` | Desativa o tracking entre frames (volta ao modo só-detecção). |
| `--report` | Gera resumo do lote via LLM usando os últimos registros do banco e sai. |
| `--report-limit N` | Nº de registros considerados pelo `--report` (padrão 30). |

### Teclas ativas na janela

| Tecla | Ação |
| --- | --- |
| `ESC` / `q` | Sair. |
| `d` | Liga/desliga exibição de todas as classes (equivale a alternar `--show-all`). |
| `s` | Salva o frame atual em `captures/AAAAMMDD_HHMMSS.jpg`. |
| `p` | Pausa/retoma o loop. |
| `l` | Pede um laudo textual (IA) do 1º boi processado. Exibe no HUD direito. |
| `i` | Pede análise visual (IA) — raça, ECC, pelagem — do 1º boi processado. |

## HUD (overlay superior esquerdo)

Mostra a cada frame:

- `FPS` — taxa de quadros.
- `Fonte` — webcam/arquivo/URL.
- `DB` — `ok`, `off` (via `--no-db`) ou `indisponivel` (sem `.env`).
- `Deteccoes (bois: N)` — total detectado e quantos são bois.
- `Escala` — cm/px calculado do `config.yaml`.
- `IA` — `ok (modelo)`, `off (--no-ia)` ou `indisponivel` quando não há chave.
- `Modo: TODAS as classes` e `PAUSADO` quando aplicável.

Quando você aperta `l` ou `i`, aparecem painéis translúcidos adicionais na lateral direita com a resposta da LLM (canto superior = laudo, canto inferior = análise visual).

## Calibrar a escala

O peso depende da escala (quantos cm vale 1 pixel). Em `config.yaml`:

```yaml
scale:
  dist_real_cm: 200   # distância real conhecida em cm (ex.: régua/cano de 2 m)
  dist_pixels: 400    # essa mesma distância, medida em pixels na imagem
```

Como ajustar na prática:

1. Coloque um objeto de tamanho conhecido na cena (ex.: cano de 2 m = 200 cm).
2. Rode o projeto e tire uma foto com `s`.
3. Abra a foto em qualquer editor e meça em pixels a extensão do objeto.
4. Ajuste `dist_real_cm` e `dist_pixels` — a HUD mostra a escala resultante.

## Filtros de processamento (`config.yaml`)

```yaml
processing:
  min_largura_m: 0.1    # descarta bois com largura menor que isto (metros)
  min_area_m2: 0.02     # descarta bois com área dorsal menor que isto (m²)
  conf: 0.25            # confiança mínima YOLO (pode ser sobrescrita com --conf)
  classes_interesse: ["cow"]
```

Use valores baixos para validar que o pipeline está rodando; aumente depois para reduzir falsos positivos.

## Estimativa de peso

`src/conversao/conversao.py` expõe três funções:

- `modelo_regressao(largura, area_dorsal)` — regressão linear simples. ⚠️ Os coeficientes `(6.15, 0.019, 70.8)` são um **placeholder didático** do TCC ArrobaDigital (IFPR), ajustados grosseiramente para Nelore adultos. Para uso real, **recalibre com dados do próprio rebanho**.
- `modelo_biometrico(comprimento, perimetro_toracico, raca)` — PT² com constante `K` por raça (bibliograficamente mais robusto; veja Santos & Boin, 1996).
- `estimar_peso(...)` — dispatcher: usa PT² se houver `comprimento` *e* `perimetro_toracico`, senão cai na regressão. Aceita `raca` como string (convertida internamente em `enum Raca`, com aliases — "Brahma" → `BRAHMAN`, "Cruzado" → `CRUZAMENTO`, "anelorado" → `NELORE`).

O peso exibido acima de cada boi é o valor **suavizado pela EMA do aggregator** (veja abaixo) a partir de 2+ amostras — não mais o valor bruto de cada frame.

## Tracking e raça dinâmica

### ByteTrack embutido

Por padrão (`--no-track` para desligar) o `main.py` usa `YoloDetector.track()`, que é um wrapper sobre `model.track(..., persist=True, tracker="bytetrack.yaml")` do Ultralytics. Cada detecção ganha um `track_id` estável entre frames. Isso resolve:

- **Reconto**: antes, cada frame podia salvar o mesmo boi no banco 12x/segundo. Agora salvamos **1x por `track_id`**, após a EMA amadurecer, com cooldown de 120 s.
- **Ruído no peso**: `src/tracking/aggregator.py` mantém uma média exponencial (EMA, α=0.3) das medidas por track. O peso estável que aparece no HUD é esse valor suavizado.

Parâmetros do `CattleAggregator` (ajustáveis em código):

| Parâmetro | Default | O que faz |
| --- | --- | --- |
| `ema_alpha` | `0.3` | Peso da amostra nova na EMA (menor = mais suave). |
| `min_amostras_para_salvar` | `8` | Só considera o track "maduro" após N frames. |
| `cooldown_salvar_s` | `120` | Intervalo mínimo entre dois salvamentos do mesmo track. |
| `expiracao_s` | `10` | Track sem update há mais que isso é descartado. |

### Raça dinâmica (`raca_provavel`)

Quando você aperta `i` (análise visual), a resposta da LLM inclui `raca_provavel`. O aggregator guarda essa raça no `track_id` e os próximos laudos e salvamentos passam a usar ela — **o `breed_focus` do `config.yaml` vira opcional**, só é usado como fallback enquanto a análise visual ainda não foi feita. Análises com `raca_provavel = "indefinido"` são ignoradas.

O `#<track_id>` é mostrado acima de cada boi junto do peso e do `n` de amostras acumuladas.

## IA (OpenRouter) — papel auxiliar

> **Princípio fundamental:** o peso é sempre calculado pelo **nosso sistema** (regressão ou PT²). A IA **nunca estima peso**, só adiciona camadas de interpretação que seriam caras/impossíveis via visão computacional clássica.

Três pontos de uso, todos opcionais:

1. **Laudo textual** (tecla `l`): recebe o peso + medidas *já calculados* e escreve 2–4 linhas em português interpretando o animal (condição, provável categoria, recomendação).
2. **Análise visual** (tecla `i`): envia o crop segmentado do boi e devolve JSON com `raca_provavel`, `confianca_raca`, `ecc` (1–5), `cor_pelagem`, `observacoes`. A raça retornada alimenta o `CattleAggregator` e passa a ser usada nos próximos laudos.
3. **Relatório do lote** (`--report`): lê os últimos N registros do banco e escreve um resumo (média, dispersão, animais fora da curva, recomendação).

### Configuração

Variáveis em `.env`:

```
API_KEY_IA=sua-chave-openrouter
IA_MODEL=openai/gpt-4o-mini
# IA_VISION_MODEL=openai/gpt-4o   # opcional: modelo separado só para tarefas com imagem
```

Modelos testados no OpenRouter (todos multimodais):

| Slug | Preço aprox. (abril/2026) | Observações |
| --- | --- | --- |
| `openai/gpt-4o-mini` | ~US$ 0.15 / US$ 0.60 por 1M tokens | **Padrão.** Equilíbrio entre qualidade e custo. Rate-limit confortável, `role=system` nativo. |
| `openai/gpt-4o` | ~US$ 2.50 / US$ 10.00 | Vision superior. Use se precisar de laudos mais detalhados. |
| `anthropic/claude-3.5-haiku` | ~US$ 0.80 / US$ 4.00 | Alternativa boa, vision decente. |
| `anthropic/claude-3.5-sonnet` | ~US$ 3.00 / US$ 15.00 | Vision excelente para ECC/conformação. |
| `google/gemini-2.5-flash` | ~US$ 0.30 / US$ 2.50 | Barato, vision nativa forte. |
| `google/gemma-4-26b-a4b-it:free` | grátis | Tier free agressivo (50 req/dia). Bom para testes. |

Só altere `IA_MODEL` no `.env` — todo o código (laudo, visão, relatório, batch) usa esse valor automaticamente. O `client.py` trata:

- **Compatibilidade `role=system`**: famílias que não aceitam (ex.: Gemma) têm o system prompt mesclado no `user` automaticamente.
- **Retry com backoff**: em `429` (rate-limit) ou `5xx`, tenta até 5 vezes com `2s → 4s → 8s → 16s → 32s`.

### Teste rápido sem câmera

```bash
# Laudo + análise visual numa imagem:
.venv/bin/python main.py --source imgs_tests/nelore.jpeg
# na janela, aperte 'l' para laudo e 'i' para análise visual

# Relatório do lote (precisa de registros no banco):
.venv/bin/python main.py --report --report-limit 20
```

Sem API_KEY_IA ou com `--no-ia`, o projeto continua rodando normalmente — só as teclas `l`/`i` e `--report` ficam indisponíveis.

## Banco de dados (opcional)

Crie um arquivo `.env` na raiz:

```
DATABASE_URL=postgresql://usuario:senha@host:5432/banco
```

Sem `.env`, o projeto continua funcionando, só não persiste. Com tracking ativo (padrão), a gravação é **1x por `track_id` depois de 8 amostras**, com cooldown de 120 s. Com `--no-track`, volta ao modo antigo: no máximo uma rodada de gravações a cada 5 s.

## Batch de imagens e ground truth

`batch_images.py` processa uma pasta de imagens sem abrir janela do OpenCV — útil para validar o pipeline em fotos conhecidas.

```bash
# Simples: roda YOLO + regressão em todas as imagens da pasta.
.venv/bin/python batch_images.py imgs_tests/ --no-ia

# Com calibração por imagem (assume que o maior boi tem 250 cm de comprimento):
.venv/bin/python batch_images.py imgs_tests/ --known-length-cm 250 --no-ia

# Comparando com pesos reais (YAML):
.venv/bin/python batch_images.py imgs_tests/ \
    --known-length-cm 250 --no-ia \
    --ground-truth imgs_tests/ground_truth.yaml

# Também pedindo um laudo AUXILIAR (raça, ECC, pelagem, observações) à IA.
# Obs: a IA NÃO calcula peso — isso é sempre do nosso sistema.
.venv/bin/python batch_images.py imgs_tests/ \
    --known-length-cm 250 \
    --ground-truth imgs_tests/ground_truth.yaml \
    --ia-analise
```

Formato do ground truth (ver `imgs_tests/ground_truth.yaml`):

```yaml
pesos:
  nelore.jpeg:     900
  nelore1200.webp: 1200
```

O resumo final imprime, por imagem: peso real vs. peso calculado pelo sistema, erro percentual, e — se `--ia-analise` estiver ligado — a raça provável devolvida pela IA. Ao fim, mostra o **MAPE do nosso sistema**.

### Lição aprendida (importante)

Em testes com 4 fotos de catálogo de Nelore PO com pesos conhecidos (900–1415 kg), a regressão linear atual mostrou **MAPE 92.8 %** — o intercepto `70.8` domina e a fórmula prevê ~81 kg para qualquer Nelore adulto visto de perfil.

**Motivo:** a regressão foi pensada para **vista dorsal** (câmera top-down em corredor/tronco) onde `largura` é *largura dorsal* real e `area_dorsal` é a projeção de cima do animal. As fotos de catálogo são de perfil lateral — aí `largura do retângulo rotacionado` = altura do corpo, medida completamente diferente. **Não é bug da fórmula, é uso fora do domínio dela.**

**Conclusão prática:**

1. Para o cenário alvo do projeto (câmera fixa em corredor/tronco, vista dorsal), a regressão pode ser recalibrada com dados próprios — veja o aviso em `src/conversao/conversao.py`.
2. Se tiver como medir **perímetro torácico** no animal ou extraí-lo da imagem com fita digital, o modelo biométrico PT² em `modelo_biometrico(...)` é bibliograficamente mais robusto (Santos & Boin, 1996).
3. Para avaliação visual complementar (raça, condição corporal, pelagem, observações), use `--ia-analise` — isso é auxílio puro ao laudo, não substitui o peso do nosso sistema.

## Modo Pessoas (experimental)

Há um pipeline **totalmente separado** em `person_demo.py` para testar o fluxo com um sujeito humano em pé na frente da câmera. Mesmo esqueleto (YOLOv8-seg + segmentação + medidas) mas filtrando a classe COCO `person` (id 0) e usando uma fórmula específica em `src/person/weight.py`.

```bash
# Atalho com calibração automática assumindo altura de 1,75 m:
make run-person

# Equivalente manual:
.venv/bin/python person_demo.py --source 0 --known-height-cm 175
```

Como a câmera raramente vem calibrada, a flag `--known-height-cm` é a maneira mais prática de acertar a escala: no primeiro frame em que exatamente **uma** pessoa é detectada, o `scale` (cm/px) é recalculado automaticamente tal que `altura_px * scale = known_height_cm`. Em seguida o peso ao vivo fica coerente. Pra recalibrar a qualquer momento (trocar de pessoa, ajustar distância), aperte `c`.

### CLI específico

| Flag | Padrão | Descrição |
| --- | --- | --- |
| `--source`, `-s` | `0` | Webcam, vídeo ou URL. |
| `--conf` | `0.35` | Confiança mínima do YOLO (mais alta que o modo boi — menos falso positivo). |
| `--scale` | — | Escala cm/px manual (sobrescreve `config.yaml`). |
| `--known-height-cm` | — | Auto-calibração pela altura real de quem está na frente. |
| `--no-track` | off | Desliga tracking (volta ao modo só-detecção). |
| `--update-every` | `2.0` | Debounce em segundos: o peso exibido só atualiza a cada N s. `0` atualiza a todo frame. |
| `--show-all` | off | Desenha todas as classes (depuração). |
| `--save-video` | — | Grava mp4 anotado. |

Teclas ativas: `ESC`/`q` sair, `p` pausar, `d` toggle show-all, `s` salvar frame em `captures/pessoa_*.jpg`, `c` recalibrar com `--known-height-cm`.

### Como a estimativa funciona

`src/person/weight.py` calcula:

1. `altura_cm` = eixo longo do retângulo rotacionado da máscara.
2. `largura_cm` = eixo curto da mesma máscara.
3. `razao = largura_cm / altura_cm` → IMC presumido por interpolação linear entre quatro pontos âncora (magro=18.5, normal=22.5, sobrepeso=30, obesidade=38).
4. `peso = IMC × altura_m²`.

O HUD mostra acima de cada pessoa: peso, faixa ± 15 %, altura, largura e IMC presumido. Se `--no-track` estiver desligado (padrão), o peso é suavizado pela EMA do mesmo `CattleAggregator` usado no modo bovino (α=0.3).

**Debounce (`--update-every`):** pra não ficar pulando números a cada frame, o peso exibido só é atualizado a cada 2 s por padrão. O YOLO continua rodando e a EMA continua acumulando em todo frame — é apenas o valor desenhado na tela que fica congelado entre duas atualizações. O contador regressivo `prox Xs` aparece na linha da faixa. Troque com `--update-every 1.0` para atualizar a cada 1 s, ou `--update-every 0` para voltar ao comportamento "todo frame".

**⚠️ Avisos importantes:**

- É **PROTÓTIPO DIDÁTICO**, não instrumento de medição. Erros reais esperados são de **15–25 %**.
- A estimativa depende de **1)** calibração correta da escala e **2)** pessoa de frente, em pé, com o corpo inteiro visível no quadro.
- Roupas largas, posições não-neutras (braços abertos, agachado, de lado) quebram o modelo.
- **Não use para decisões médicas ou nutricionais.** Nada substitui balança.

### O que *não* é feito pelo modo Pessoas (intencional)

- Não persiste no banco (é uma demo pura; DB desativado).
- Não chama a IA (laudo/análise visual do `main.py` é específico de bovinos).
- Não acumula histórico em arquivo — se quiser, basta `--save-video`.

Se quiser persistir peso humano no DB depois, o caminho é replicar a lógica de `deve_salvar` do `main.py` com `min_amostras_para_salvar` menor (tipo 5) no aggregator do demo.

## Testes

```bash
make test-unit          # pytest -q
# ou direto:
.venv/bin/python -m pytest -q
```

Cobertura atual (`tests/`):

- `test_conversao.py` — regressão, PT², mapeamento de nomes de raça, dispatcher `estimar_peso`.
- `test_measurements.py` — `calculate_scale` e `extract_measurements` sobre máscaras sintéticas.
- `test_visao_json.py` — parser `_extract_json` com/sem fences, JSON inválido, `_as_float` com limites.
- `test_aggregator.py` — EMA, política de `deve_salvar`, raça dinâmica, expiração de tracks.
- `test_person_weight.py` — fórmula do modo Pessoas (pontos âncora, clamp, validação).
