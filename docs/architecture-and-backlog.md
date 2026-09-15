# fin-app — ідеї, архітектура, беклог

> Цей файл — головний довідник проєкту. Тут накопичується все, що обговорюється в "основному" чаті (ідеї, архітектурні рішення, беклог). Окремі модулі/аспекти розбираються в інших чатах, повʼязаних з fin-app, — при потребі підвантажуй цей файл як контекст.
>
> Останнє оновлення: 2026-09-12.

## 1. Ідея проєкту

Аналітика витрат і доходів для оптимізації особистих фінансів:

- Отримання даних по категоріях, датах тощо.
- Базова аналітика доходів/витрат (графіки, тренди).
- ML-модель для глибшого аналізу і передбачення.
- LLM, яка на основі визначеної користувачем цілі видає рекомендації по витратах.

## 2. Поточний стан репозиторію (на момент старту)

Стандартний Databricks Asset Bundle (`default-python` темплейт), ще без реального фінансового коду:

- `databricks.yml` — bundle `finapp`, workspace `adb-7405617288273171` (Azure). Таргети `dev` (mode: development, catalog=`dev`/schema=`default`) і `prod` (catalog=`fin_app_7405617288273171`/schema=`prod`, root_path у `/Workspace/Users/ohryc@softserveinc.com/...`, CAN_MANAGE лише для ohryc).
- `resources/finapp_etl.pipeline.yml` — DLT-пайплайн `finapp_etl`, читає трансформації з `src/finapp_etl/transformations`.
- `resources/sample_job.job.yml` — `sample_job` (notebook → wheel task `main` → refresh pipeline), кластер `16.4.x-scala2.12` / `Standard_D3_v2`.
- `src/finapp/` — пакет з `main.py` (entry point) і `taxis.py` (sample: читає `samples.nyctaxi.trips`).
- `src/finapp_etl/transformations/` — sample DLT-датасети (`sample_trips_finapp`, `sample_zones_finapp`) — це темплейтні заглушки, підлягають заміні на реальні фінансові трансформації.
- `tests/` — pytest + Databricks Connect фікстури.
- Одна незакомічена зміна в `databricks.yml` (dev: catalog/schema `fin_app_7405617288273171`/`dev` → `dev`/`default`) — ще не в git.

## 3. Повна архітектура (цільова)

### 3.1 Шари

```
Джерела → Ingestion → Lake (ADLS Gen2, medallion, Unity Catalog) → категоризація/ML → Аналітика (Gold) → LLM-рекомендації → Інтерфейс
```

Все розширюється в межах наявного DAB: нові ресурси в `resources/*.yml`, ті самі `dev`/`prod` таргети.

### 3.2 Джерела даних та ingestion

- **Ручне введення** — через інтерфейс (див. розділ 4), пише напряму в Unity Catalog таблицю (Delta), без проміжного ADLS-файлу.
- **Monobank API** — реалістичний варіант: особистий токен на api.monobank.ua, ендпоінт `personal/statement/{account}/{from}/{to}` (до 500 транзакцій за раз, вікно максимум 31 доба, ліміт 1 запит/60 сек). Підтримує і webhook на нові транзакції (кращий за polling). MCC/опис мерчанта — основа для автокатегоризації.
- **Privat24 API** — теж технічно доступний для фізосіб через api.privatbank.ua (read-only токен на виписку), але менш стабільний/документований. Закладати як опціональний конектор з тим самим інтерфейсом, що й Mono, щоб підключати/відключати без переробки решти пайплайна.
- Ingestion банківських джерел — окремий Databricks Job (`bank_ingestion_job`), токени — тільки в Databricks Secret Scope, сирі відповіді льються в Bronze.

### 3.3 Модель даних (medallion + Unity Catalog)

- **Bronze** — сирі дані per джерело: `bronze.mono_raw`, `bronze.privat_raw`, `bronze.manual_raw`. Без трансформацій, для трасованості.
- **Silver** — єдина схема транзакцій: `silver.transactions` (transaction_id, account_id, user_id, date, amount, currency, direction income/expense, merchant_raw, source, category_id nullable, category_source: manual/rule/ml, confidence), `silver.accounts`, `silver.categories` (ієрархія група→підкатегорія), `silver.category_rules` (ключові слова/MCC → категорія).
- **Gold** — агрегати під аналітику й фічі для ML: `gold.spend_by_category_month`, `gold.income_vs_expense_trend`, `gold.recurring_payments` (підписки), `gold.user_goals` (ціль: сума/категорія/період), feature-таблиці для моделей.
- Мапиться на наявний `finapp_etl` DLT-пайплайн: нові `@dp.table` трансформації замість sample nyctaxi-датасетів.

### 3.4 Категоризація

Триступенева логіка:

1. Ручний запис — категорія задана користувачем напряму ("золоті" мітки).
2. Rule-based — за MCC-кодом/ключовими словами в описі мерчанта (швидко, покриває 60–80%).
3. ML-класифікатор — навчений на накопичених мітках (TF-IDF/embeddings + gradient boosting, або NLP-модель). Для холодного старту, поки мало даних, — тимчасово LLM у zero/few-shot режимі, з поступовим переходом на власну ML-модель.

Feedback-цикл: виправлення користувачем автокатегорії → нова мітка в Silver → періодичний retraining.

### 3.5 Аналітичний шар

Gold-таблиці → Databricks SQL Dashboard або Power BI (в організації вже стандарт) поверх Unity Catalog. Розрізи: доходи vs витрати по місяцях, розподіл по категоріях, тренд/сезонність, топ-мерчанти, факт vs бюджет, регулярні платежі.

### 3.6 ML-компонент

Окремий тренувальний job (MLflow + Unity Catalog Model Registry):

- Прогноз витрат по категоріях (лаг-фічі + gradient boosting, або Prophet/ARIMA для сезонності).
- Anomaly detection — нетипові транзакції.
- Кластеризація патернів витрат.

Результати пишуться в Gold і живлять LLM-шар.

### 3.7 LLM-рекомендаційний шар

Комбінований підхід (обрано користувачем): ML дає структуровані сигнали (прогноз, аномалії, %-відхилення), LLM на вході отримує ці сигнали + явну ціль користувача → генерує персоналізовану текстову рекомендацію.

Governance: фінансові персональні дані → краще Databricks Foundation Model API (всередині workspace) замість зовнішнього LLM API, щоб дані не покидали контрольований периметр. Оформлюється як MLflow pyfunc-модель / serving endpoint.

### 3.8 Безпека і governance

- Токени Mono/Privat24 — лише в Databricks Secret Scopes.
- Unity Catalog: розмежування доступу Bronze/Silver (обмежено) vs Gold (ширше).
- Пізніше, якщо буде кілька користувачів — row-level security за `user_id`.

## 4. Інтерфейс (перша фаза розробки)

Обрано: **Databricks App на Streamlit**, як ресурс того ж бандла (поруч із `finapp_etl.pipeline.yml`, `sample_job.job.yml`).

- Databricks Apps: контейнеризований застосунок на serverless compute у workspace; підтримує Streamlit/Dash/Gradio (Python) або Node (React/Angular/Svelte/Express); вбудована автентифікація (OIDC/OAuth 2.0, SSO) і governed-доступ до Unity Catalog / SQL Warehouse.
- **Форма ручного вводу** — пише рядок напряму в `silver.manual_entries` (або `silver.transactions` з `source=manual`) через Databricks SQL Connector; права INSERT — окремому service principal застосунку через UC grants.
- **Графіки** — та ж Streamlit-сторінка, запити до Gold через SQL Warehouse, рендер через Plotly/Altair.
- **ML-прогноз/інсайти** — модель тренується offline (Databricks Job + MLflow, реєстрація в UC Model Registry); застосунок лише звертається до Model Serving endpoint або читає готову таблицю прогнозів із Gold.
- Деплой — той самий `databricks bundle deploy` в `dev`/`prod`.

**Розглянуті альтернативи:**
- Окремий бекенд (Azure App Service/Function + кастомний фронтенд) — коли потрібен повністю кастомний UI/мобільний застосунок; мінус — втрата вбудованої auth/governance.
- Power BI (перегляд) + Power Apps (форма) — знайомий інструмент перегляду, але два окремі інструменти замість одного, складніше звʼязати з ML у реальному часі.
- Databricks Lakebase (managed Postgres) — якщо з часом буде багато дрібних транзакційних записів (OLTP-патерн); на старті зайва складність.

## 5. Беклог / послідовність реалізації

1. Модель даних + Bronze/Silver схема + ingestion ручного вводу (найпростіший шлях отримати перші реальні дані).
2. Rule-based категоризація + базові Gold-агрегації + перший дашборд.
3. Databricks App (Streamlit): форма вводу + базові графіки.
4. Підключення Monobank API (простіший з двох), потім опціонально Privat24.
5. ML-класифікатор категорій (коли назбирається розмічених даних) + прогнозна модель (forecast/anomaly).
6. Інтеграція ML-прогнозів у застосунок (сторінка "глибокий аналіз").
7. LLM-рекомендаційний шар поверх готової аналітики та прогнозів (ціль користувача → рекомендація).

## 6. Відкриті питання / рішення, що очікують

- Точний перелік Unity Catalog grants для service principal застосунку (SELECT на Gold/Silver, INSERT на manual entries) — деталізувати перед імплементацією.
- Структура файлів Databricks App (`app.py`, `app.yaml`, `resources/finapp_app.app.yml`) — розписати перед створенням.
- Чи потрібен окремий `user_id`/мультикористувацький режим з самого початку, чи це single-user проєкт на старті.
- Остаточне рішення по Privat24 (підключати одразу чи залишити на потім через нестабільність API).
