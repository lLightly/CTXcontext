### 4. `README.md`

```markdown
# ctxforge (`ctx`)

**`ctxforge`** — локальный open-source компилятор контекста, индексатор AST-скелетов и реестр проектных решений с механизмом **Truth Governance** для долгосрочной разработки с LLM (ориентирован на бесплатный веб-интерфейс **Google AI Studio**, локальные модели через **Ollama** и бесплатные тарифы Gemini Flash API — **без платных API и обязательной мультиагентности**).

- **Нулевые внешние зависимости в ядре**: только стандартная библиотека **Python 3.11+**.
- **Защита от контекстной деградации (Context Rot)**: устаревшие (`SUPERSEDED`) и отвергнутые (`PITFALL`) решения не попадают в контекст как активные рекомендации, а выносятся в жёсткий блок `<rejected_approaches>`.
- **Борьба с «Lost-in-the-Middle»**: неукоснительные инварианты и задача размещаются в начале капсулы, а компактный машиночитаемый протокол `<ctx_protocol>` — в конце.
- **Tier-1 / Tier-2 / Tier-3 сжатие кода**: целевые файлы идут целиком, смежные по графу импортов и Git co-change сжимаются до AST-сигнатур и docstrings, остальной проект — в компактное дерево путей.
- **Reality-Drift Doctor**: проверка соответствия памяти `.ctx/` реальным файлам, символам и SHA-256 хэшам без вызова LLM.

---

## Установка

Требуется **Python 3.11+** (опционально `git` для анализа co-change).

```bash
git clone <repo-url> ctxforge
cd ctxforge
pip install -e . --no-build-isolation
```

После установки доступна команда `ctx`. Также можно запускать напрямую без установки:

```bash
python -m ctxforge.cli --help
```

---

## Быстрый старт за 3 минуты на реальном проекте (Google AI Studio)

Ниже показан полный рабочий цикл на реальном проекте (например, CLI-инструменте вроде **SDD Token Optimizer**):

### Шаг 1. Инициализация памяти в корне проекта

```bash
cd /path/to/sdd-token-optimizer
ctx init --name "sdd-token-optimizer" --vision "CLI-оптимизатор промптов и токен-бюджета для SDD"
```

Создаётся папка `.ctx/` (`project.json`, `cards.jsonl`, `tasks.json`, `sessions.jsonl`, `.gitignore` для `cache/` и `out/`).

### Шаг 2. Day-0 оцифровка проекта (Bootstrap)

Сгенерируйте стартовый промпт из вашего ТЗ/описания проекта (`brief.md`) и AST-скелетов уже существующего кода:

```bash
ctx bootstrap brief.md --budget 12000
```

Промпт сохранится в `.ctx/out/bootstrap_prompt.md` и автоматически скопируется в буфер обмена.
1. Откройте **Google AI Studio**, вставьте промпт (`Ctrl+V`) и отправьте.
2. Скопируйте ответ модели (содержащий `<ctx_bootstrap>...</ctx_bootstrap>`).
3. Импортируйте его в `.ctx/`:

```bash
ctx ingest --bootstrap
```

### Шаг 3. Компиляция капсулы под задачу

Посмотрите доступные задачи и скомпилируйте токен-бюджетированную капсулу:

```bash
ctx task list
ctx compile TASK-001 --budget 8000
```

`ctx compile`:
- в Tier-1 положит целевые файлы целиком,
- в Tier-2 добавит AST-сигнатуры зависимостей,
- отфильтрует только активные `INV` / `ADR` / `PITFALL`,
- скопирует капсулу в буфер обмена и покажет реальную экономию токенов (`Full: ... -> Capsule: ... (-78%)`).

Вставьте капсулу в новый чат **Google AI Studio** и решайте задачу.

### Шаг 4. Обратная связь (`<ctx_delta>`) и handoff

В конце ответа модель вернёт компактный блок `<ctx_delta>`. Скопируйте ответ модели и примените изменения с предпросмотром diff и защитой `immutable`-инвариантов:

```bash
ctx ingest
```

Если диалог затянулся и нужно срочно переехать в новый чат:

```bash
ctx handoff TASK-001
```

Команда сгенерирует ультра-короткий (~80 токенов) промпт для выгрузки `<ctx_delta>`.

*(Опционально: если вы забыли попросить `<ctx_delta>` и скопировали обычный длинный чат, можно сжать его через бесплатный Gemini Flash или локальную Ollama: `ctx ingest --extract-via ollama --model qwen3` или `GEMINI_API_KEY=... ctx ingest --extract-via gemini`.)*

### Шаг 5. Контроль дрейфа реальности (`doctor` и `status`)

```bash
ctx doctor
ctx status
```

`ctx doctor` без LLM проверит битые якоря (`BROKEN_ANCHOR`), устаревшие хэши изменённых файлов (`NEEDS_REVIEW`), циклы в DAG задач и перераздувание `always_include` (>10 карточек или >1500 токенов). После ручного ревью дрейфа обновите хэши:

```bash
ctx doctor --refresh-hashes
```

---

## Справочник основных команд

| Команда | Назначение |
|---|---|
| `ctx init [--name] [--vision]` | Инициализация `.ctx/` и первичный AST/Git индекс репозитория |
| `ctx bootstrap <brief_or_dash> [--budget 12000]` | Генерация стартового промпта для Day-0 оцифровки |
| `ctx task add --title "..." [--anchors ...] [--deps ...] [--ac "..."]` | Добавить задачу в DAG |
| `ctx task list [--status todo\|in_progress\|done\|all]` | Список задач |
| `ctx task done <task_id>` | Отметить задачу выполненной |
| `ctx card add --kind INV\|ADR\|PITFALL --title "..." --statement "..." --rationale "..." [--immutable] [--always]` | Добавить карточку знания |
| `ctx card list [--all]` | Показать активные карточки (или все, включая `SUPERSEDED`) |
| `ctx compile [<task_id>] [--query "..."] [--files ...] [--budget 8000]` | Скомпилировать капсулу контекста под задачу/запрос |
| `ctx handoff [<task_id>]` | Короткий промпт для экстренной выгрузки `<ctx_delta>` |
| `ctx ingest [--file PATH] [--bootstrap] [--auto] [--force-immutable]` | Применить `<ctx_delta>` / `<ctx_bootstrap>` с предпросмотром diff |
| `ctx doctor [--refresh-hashes]` | Диагностика битых якорей, дрейфа хэшей, циклов DAG и шума |
| `ctx status` | Краткий дашборд состояния памяти проекта и последней капсулы |
| `ctx bench init-sample \| prepare \| score \| run` | A/B-эксперимент сравнения стратегий контекста |

---

## Проведение эксперимента A/B (`ctx bench`) и интерпретация отчёта

Чтобы экспериментально сравнить 3 подхода на одинаковых задачах:
1. `baseline_full_dump` — передача всего репозитория и всех карточек без фильтрации;
2. `baseline_long_chat` — имитация «грязной» длинной истории чата с устаревшими/отвергнутыми идеями;
3. `ctxforge_compiled` — скомпилированная капсула `ctxforge`.

### Вариант A: Быстрая проверка за 1 минуту на встроенной песочнице

```bash
ctx bench init-sample --dir /tmp/ctx-sandbox
ctx --root /tmp/ctx-sandbox bench prepare --tasks TASK-001,TASK-002
```

### Вариант B: Ручной бесплатный A/B в веб-интерфейсе Google AI Studio

1. Подготовьте пакеты промптов на своём проекте:
   ```bash
   ctx bench prepare --tasks TASK-001,TASK-002 --out-dir .ctx/bench
   ```
   Команда выведет таблицу **Input Token Footprint** и **Signal-to-Noise Ratio (SNR)** и сохранит промпты в `.ctx/bench/prompts/<TASK>__<strategy>.md`.
2. Откройте поочерёдно промпты в новых чатах Google AI Studio и сохраните ответы в `.ctx/bench/responses/<TASK>__<strategy>.md`.
3. Оцените ответы и сгенерируйте отчёт:
   ```bash
   ctx bench score --dir .ctx/bench --test-cmd "python -m unittest discover -s tests"
   ```

### Вариант C: Полностью автоматический прогон (Ollama или Gemini Free API)

```bash
# Локально через Ollama:
ctx bench run --provider ollama --model qwen3 --tasks TASK-001,TASK-002

# Или через бесплатный Gemini Flash API:
export GEMINI_API_KEY="your-free-ai-studio-key"
ctx bench run --provider gemini --model gemini-2.5-flash --tasks TASK-001,TASK-002
```

### Как интерпретировать `.ctx/bench/benchmark_report.md`

Отчёт содержит сводную таблицу, ASCII-график и проверку трёх гипотез:
- **H1 (Token Footprint & SNR)**: `ctxforge_compiled` должен снижать входной объём токенов (обычно на 65–85% относительно полного дампа) при максимальном отношении сигнал/шум (**SNR**).
- **H2 (Truth Governance & Relapse Prevention)**: количество рецидивов в отвергнутые идеи (`Relapses` по `PITFALL`/`SUPERSEDED`) и нарушений инвариантов (`Adherence`) в `ctxforge_compiled` по сравнению с `baseline_long_chat`.
- **H3 (Grounding & Anchor Accuracy)**: доля корректных ссылок на существующие файлы и символы (`Anchor Acc`), показывающая снижение галлюцинаций путей.
```

---

### Установка и запуск всех тестов + первого реального прогона

1. **Установка (опционально, в editable-режиме без изоляции сборки или стандартно):**

```bash
pip install -e . --no-build-isolation
```

2. **Запуск полного набора всех тестов (Таски 1–6):**

```bash
python -m unittest discover -s tests -v
```

3. **Первый реальный прогон A/B-стенда на встроенной песочнице (за 1 минуту, не затрагивая рабочий код):**

```bash
ctx bench init-sample --dir /tmp/ctx-sandbox
ctx --root /tmp/ctx-sandbox bench prepare --tasks TASK-001,TASK-002
```

После этого можно вставить промпты из `/tmp/ctx-sandbox/.ctx/bench/prompts/` в Google AI Studio, сохранить ответы в `/tmp/ctx-sandbox/.ctx/bench/responses/` и запустить оценку:

```bash
ctx --root /tmp/ctx-sandbox bench score --test-cmd "python -m unittest discover -s tests"
```

Или при запущенной локальной модели Ollama:

```bash
ctx --root /tmp/ctx-sandbox bench run --provider ollama --model qwen3 --tasks TASK-001,TASK-002 --test-cmd "python -m unittest discover -s tests"
```