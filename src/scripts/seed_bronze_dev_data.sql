-- Dev seed data for Bronze layer (fin-app)
-- Схема свідомо мінімальна й наближена до реальних payload-ів джерел (Mono/Privat/manual),
-- щоб трансформації Silver можна було розробляти вже зараз (docs/architecture-and-backlog.md, §3.3).
-- ПРОПОЗИЦІЯ схеми bronze.* — ще не зафіксована в architecture-and-backlog.md, потребує підтвердження.
--
-- Каталог/схема нижче ("dev"/"bronze") — приклад під dev-таргет. databricks.yml зараз має
-- одну змінну schema (dev -> default, prod -> prod) на весь пайплайн; для medallion (bronze/silver/gold)
-- знадобляться або окремі схеми per шар (як тут), або префіксовані назви таблиць в одній схемі —
-- це відкрите архітектурне питання, ще не зафіксоване в §6 backlog-документа.

USE CATALOG dev;
CREATE SCHEMA IF NOT EXISTS bronze MANAGED LOCATION 'abfss://dev@ohrycfinapp.dfs.core.windows.net/bronze';

CREATE TABLE IF NOT EXISTS bronze.mono_raw (
  account_id STRING,
  transaction_id STRING,
  time_unix BIGINT,
  description STRING,
  mcc INT,
  original_mcc INT,
  amount BIGINT,
  operation_amount BIGINT,
  currency_code INT,
  commission_rate BIGINT,
  cashback_amount BIGINT,
  balance BIGINT,
  comment STRING,
  receipt_id STRING,
  counter_edrpou STRING,
  counter_iban STRING,
  counter_name STRING,
  _ingested_at TIMESTAMP,
  _source STRING
) USING DELTA;

CREATE TABLE IF NOT EXISTS bronze.privat_raw (
  account_id STRING,
  transaction_id STRING,
  trandate STRING,
  trantime STRING,
  amount DECIMAL(18,2),
  currency STRING,
  description STRING,
  card_number_masked STRING,
  balance DECIMAL(18,2),
  terminal STRING,
  _ingested_at TIMESTAMP,
  _source STRING
) USING DELTA;

CREATE TABLE IF NOT EXISTS bronze.manual_raw (
  entry_id STRING,
  user_id STRING,
  entry_ts TIMESTAMP,
  amount DECIMAL(18,2),
  currency STRING,
  direction STRING,
  merchant_raw STRING,
  category_hint STRING,
  note STRING,
  _source STRING
) USING DELTA;

-- 20 рядків: bronze.mono_raw (amount/balance у копійках, від'ємні = витрата)
INSERT INTO bronze.mono_raw VALUES
('acc-mono-1', 'mono-tx-0001', unix_timestamp(TIMESTAMP'2026-07-15 16:47:00'), 'Glovo', 5812, 5812, -69196, -69196, 980, 0, 691, 3130804, NULL, 'rcpt-mono-tx-0001', NULL, NULL, NULL, TIMESTAMP'2026-07-15 16:49:00', 'monobank'),
('acc-mono-1', 'mono-tx-0002', unix_timestamp(TIMESTAMP'2026-07-29 18:47:00'), 'Розетка', 5732, 5732, -182392, -182392, 980, 0, 1823, 2948412, NULL, 'rcpt-mono-tx-0002', NULL, NULL, NULL, TIMESTAMP'2026-07-29 18:49:00', 'monobank'),
('acc-mono-1', 'mono-tx-0003', unix_timestamp(TIMESTAMP'2026-09-08 17:37:00'), 'АЗС WOG', 5541, 5541, -13331, -13331, 980, 0, 133, 2935081, NULL, 'rcpt-mono-tx-0003', NULL, NULL, NULL, TIMESTAMP'2026-09-08 17:39:00', 'monobank'),
('acc-mono-1', 'mono-tx-0004', unix_timestamp(TIMESTAMP'2026-07-04 17:13:00'), 'Vodafone Україна', 4814, 4814, -137475, -137475, 980, 0, 1374, 2797606, NULL, 'rcpt-mono-tx-0004', NULL, NULL, NULL, TIMESTAMP'2026-07-04 17:15:00', 'monobank'),
('acc-mono-1', 'mono-tx-0005', unix_timestamp(TIMESTAMP'2026-07-05 00:12:00'), 'Netflix', 5815, 5815, -114974, -114974, 980, 0, 1149, 2682632, NULL, NULL, NULL, NULL, NULL, TIMESTAMP'2026-07-05 00:14:00', 'monobank'),
('acc-mono-1', 'mono-tx-0006', unix_timestamp(TIMESTAMP'2026-07-29 23:37:00'), 'Glovo', 5812, 5812, -217187, -217187, 980, 0, 2171, 2465445, NULL, 'rcpt-mono-tx-0006', NULL, NULL, NULL, TIMESTAMP'2026-07-29 23:39:00', 'monobank'),
('acc-mono-1', 'mono-tx-0007', unix_timestamp(TIMESTAMP'2026-07-02 04:51:00'), 'Укрзалізниця', 4112, 4112, -188013, -188013, 980, 0, 1880, 2277432, NULL, 'rcpt-mono-tx-0007', NULL, NULL, NULL, TIMESTAMP'2026-07-02 04:53:00', 'monobank'),
('acc-mono-1', 'mono-tx-0008', unix_timestamp(TIMESTAMP'2026-08-24 21:17:00'), 'Comfy', 5732, 5732, -61443, -61443, 980, 0, 614, 2215989, NULL, 'rcpt-mono-tx-0008', NULL, NULL, NULL, TIMESTAMP'2026-08-24 21:19:00', 'monobank'),
('acc-mono-1', 'mono-tx-0009', unix_timestamp(TIMESTAMP'2026-08-13 17:05:00'), 'Кешбек Monobank', 6012, 6012, 12500, 12500, 980, 0, 0, 2228489, NULL, NULL, NULL, NULL, 'Ivanenko FOP', TIMESTAMP'2026-08-13 17:07:00', 'monobank'),
('acc-mono-1', 'mono-tx-0010', unix_timestamp(TIMESTAMP'2026-07-13 21:54:00'), 'Нова Пошта', 4215, 4215, -163263, -163263, 980, 0, 1632, 2065226, NULL, NULL, NULL, NULL, NULL, TIMESTAMP'2026-07-13 21:56:00', 'monobank'),
('acc-mono-1', 'mono-tx-0011', unix_timestamp(TIMESTAMP'2026-08-04 04:02:00'), 'Аптека Подорожник', 5912, 5912, -145568, -145568, 980, 0, 1455, 1919658, NULL, 'rcpt-mono-tx-0011', NULL, NULL, NULL, TIMESTAMP'2026-08-04 04:04:00', 'monobank'),
('acc-mono-1', 'mono-tx-0012', unix_timestamp(TIMESTAMP'2026-07-17 06:24:00'), 'Novus', 5411, 5411, -149714, -149714, 980, 0, 1497, 1769944, NULL, 'rcpt-mono-tx-0012', NULL, NULL, NULL, TIMESTAMP'2026-07-17 06:26:00', 'monobank'),
('acc-mono-1', 'mono-tx-0013', unix_timestamp(TIMESTAMP'2026-08-08 05:40:00'), 'Нова Пошта', 4215, 4215, -156349, -156349, 980, 0, 1563, 1613595, NULL, 'rcpt-mono-tx-0013', NULL, NULL, NULL, TIMESTAMP'2026-08-08 05:42:00', 'monobank'),
('acc-mono-1', 'mono-tx-0014', unix_timestamp(TIMESTAMP'2026-07-26 03:04:00'), 'АТБ', 5411, 5411, -178346, -178346, 980, 0, 1783, 1435249, NULL, 'rcpt-mono-tx-0014', NULL, NULL, NULL, TIMESTAMP'2026-07-26 03:06:00', 'monobank'),
('acc-mono-1', 'mono-tx-0015', unix_timestamp(TIMESTAMP'2026-07-31 04:18:00'), 'Novus', 5411, 5411, -229216, -229216, 980, 0, 2292, 1206033, NULL, NULL, NULL, NULL, NULL, TIMESTAMP'2026-07-31 04:20:00', 'monobank'),
('acc-mono-1', 'mono-tx-0016', unix_timestamp(TIMESTAMP'2026-07-31 05:06:00'), 'АЗС ОККО', 5541, 5541, -77869, -77869, 980, 0, 778, 1128164, NULL, 'rcpt-mono-tx-0016', NULL, NULL, NULL, TIMESTAMP'2026-07-31 05:08:00', 'monobank'),
('acc-mono-1', 'mono-tx-0017', unix_timestamp(TIMESTAMP'2026-08-29 02:53:00'), 'Нова Пошта', 4215, 4215, -47638, -47638, 980, 0, 476, 1080526, NULL, 'rcpt-mono-tx-0017', NULL, NULL, NULL, TIMESTAMP'2026-08-29 02:55:00', 'monobank'),
('acc-mono-1', 'mono-tx-0018', unix_timestamp(TIMESTAMP'2026-08-17 21:13:00'), 'Повернення за товар', 5732, 5732, 45000, 45000, 980, 0, 0, 1125526, NULL, NULL, NULL, NULL, 'Ivanenko FOP', TIMESTAMP'2026-08-17 21:15:00', 'monobank'),
('acc-mono-1', 'mono-tx-0019', unix_timestamp(TIMESTAMP'2026-08-05 03:59:00'), 'Novus', 5411, 5411, -164680, -164680, 980, 0, 1646, 960846, NULL, 'rcpt-mono-tx-0019', NULL, NULL, NULL, TIMESTAMP'2026-08-05 04:01:00', 'monobank'),
('acc-mono-1', 'mono-tx-0020', unix_timestamp(TIMESTAMP'2026-07-23 00:46:00'), 'Vodafone Україна', 4814, 4814, -47834, -47834, 980, 0, 478, 913012, NULL, NULL, NULL, NULL, NULL, TIMESTAMP'2026-07-23 00:48:00', 'monobank');

-- 20 рядків: bronze.privat_raw (amount у гривнях, від'ємні = витрата)
INSERT INTO bronze.privat_raw VALUES
('acc-privat-1', 'privat-tx-0001', '2026-08-29', '22:17:00', -625.7, 'UAH', 'Netflix', '4149**** **** 5521', 14374.3, 'POS Kyiv', TIMESTAMP'2026-08-29 22:20:00', 'privat24'),
('acc-privat-1', 'privat-tx-0002', '2026-08-12', '05:49:00', -650.43, 'UAH', 'АТБ', '4149**** **** 5521', 13723.87, 'POS Kyiv', TIMESTAMP'2026-08-12 05:52:00', 'privat24'),
('acc-privat-1', 'privat-tx-0003', '2026-07-06', '04:20:00', -751.86, 'UAH', 'АЗС ОККО', '4149**** **** 5521', 12972.01, 'POS Kyiv', TIMESTAMP'2026-07-06 04:23:00', 'privat24'),
('acc-privat-1', 'privat-tx-0004', '2026-07-09', '19:58:00', -1931.96, 'UAH', 'Спортмастер', '4149**** **** 5521', 11040.05, 'POS Kyiv', TIMESTAMP'2026-07-09 20:01:00', 'privat24'),
('acc-privat-1', 'privat-tx-0005', '2026-08-10', '19:41:00', -1087.12, 'UAH', 'McDonald's', '4149**** **** 5521', 9952.93, 'POS Kyiv', TIMESTAMP'2026-08-10 19:44:00', 'privat24'),
('acc-privat-1', 'privat-tx-0006', '2026-08-28', '18:16:00', -696.51, 'UAH', 'Comfy', '4149**** **** 5521', 9256.42, 'POS Kyiv', TIMESTAMP'2026-08-28 18:19:00', 'privat24'),
('acc-privat-1', 'privat-tx-0007', '2026-09-11', '00:16:00', -1173.11, 'UAH', 'Спортмастер', '4149**** **** 5521', 8083.31, 'POS Kyiv', TIMESTAMP'2026-09-11 00:19:00', 'privat24'),
('acc-privat-1', 'privat-tx-0008', '2026-08-21', '21:14:00', 18500.0, 'UAH', 'Зарплата - ФОП Іваненко', '4149**** **** 5521', 26583.31, 'POS Kyiv', TIMESTAMP'2026-08-21 21:17:00', 'privat24'),
('acc-privat-1', 'privat-tx-0009', '2026-09-04', '23:05:00', -337.43, 'UAH', 'АТБ', '4149**** **** 5521', 26245.88, 'POS Kyiv', TIMESTAMP'2026-09-04 23:08:00', 'privat24'),
('acc-privat-1', 'privat-tx-0010', '2026-07-21', '02:10:00', -1613.45, 'UAH', 'АЗС WOG', '4149**** **** 5521', 24632.43, 'POS Kyiv', TIMESTAMP'2026-07-21 02:13:00', 'privat24'),
('acc-privat-1', 'privat-tx-0011', '2026-07-09', '22:24:00', -1437.04, 'UAH', 'Аптека Подорожник', '4149**** **** 5521', 23195.39, 'POS Kyiv', TIMESTAMP'2026-07-09 22:27:00', 'privat24'),
('acc-privat-1', 'privat-tx-0012', '2026-08-03', '00:55:00', -1833.32, 'UAH', 'Сільпо', '4149**** **** 5521', 21362.07, 'POS Kyiv', TIMESTAMP'2026-08-03 00:58:00', 'privat24'),
('acc-privat-1', 'privat-tx-0013', '2026-07-16', '02:56:00', -2018.38, 'UAH', 'Netflix', '4149**** **** 5521', 19343.69, 'POS Kyiv', TIMESTAMP'2026-07-16 02:59:00', 'privat24'),
('acc-privat-1', 'privat-tx-0014', '2026-08-05', '04:41:00', -342.42, 'UAH', 'Епіцентр', '4149**** **** 5521', 19001.27, 'POS Kyiv', TIMESTAMP'2026-08-05 04:44:00', 'privat24'),
('acc-privat-1', 'privat-tx-0015', '2026-08-07', '22:10:00', -58.5, 'UAH', 'Аптека Подорожник', '4149**** **** 5521', 18942.77, 'POS Kyiv', TIMESTAMP'2026-08-07 22:13:00', 'privat24'),
('acc-privat-1', 'privat-tx-0016', '2026-08-04', '00:48:00', 18500.0, 'UAH', 'Зарплата - ФОП Іваненко', '4149**** **** 5521', 37442.77, 'POS Kyiv', TIMESTAMP'2026-08-04 00:51:00', 'privat24'),
('acc-privat-1', 'privat-tx-0017', '2026-09-04', '06:06:00', -1724.96, 'UAH', 'METRO', '4149**** **** 5521', 35717.81, 'POS Kyiv', TIMESTAMP'2026-09-04 06:09:00', 'privat24'),
('acc-privat-1', 'privat-tx-0018', '2026-09-04', '01:12:00', -1030.19, 'UAH', 'Comfy', '4149**** **** 5521', 34687.62, 'POS Kyiv', TIMESTAMP'2026-09-04 01:15:00', 'privat24'),
('acc-privat-1', 'privat-tx-0019', '2026-07-22', '00:49:00', -51.49, 'UAH', 'Ощадбанк - комуналка', '4149**** **** 5521', 34636.13, 'POS Kyiv', TIMESTAMP'2026-07-22 00:52:00', 'privat24'),
('acc-privat-1', 'privat-tx-0020', '2026-08-11', '23:01:00', -1001.53, 'UAH', 'Розетка', '4149**** **** 5521', 33634.6, 'POS Kyiv', TIMESTAMP'2026-08-11 23:04:00', 'privat24');

-- 10 рядків: bronze.manual_raw (ручний ввід через майбутній Databricks App)
INSERT INTO bronze.manual_raw VALUES
('manual-0001', 'user-oleksandr', TIMESTAMP'2026-08-13 19:00:00', 387.0, 'UAH', 'expense', 'Таксі', 'Транспорт', NULL, 'manual'),
('manual-0002', 'user-oleksandr', TIMESTAMP'2026-07-16 03:00:00', 2140.0, 'UAH', 'expense', 'Кафе', 'Комунальні', NULL, 'manual'),
('manual-0003', 'user-oleksandr', TIMESTAMP'2026-07-22 02:00:00', 2096.0, 'UAH', 'expense', 'Кіно', 'Комунальні', 'готівкою', 'manual'),
('manual-0004', 'user-oleksandr', TIMESTAMP'2026-08-08 00:00:00', 25000.0, 'UAH', 'income', 'Зарплата', 'Зарплата', NULL, 'manual'),
('manual-0005', 'user-oleksandr', TIMESTAMP'2026-07-31 03:00:00', 1426.0, 'UAH', 'expense', 'Магазин одягу', 'Фріланс', NULL, 'manual'),
('manual-0006', 'user-oleksandr', TIMESTAMP'2026-08-31 06:00:00', 2269.0, 'UAH', 'expense', 'Магазин одягу', 'Транспорт', 'готівкою', 'manual'),
('manual-0007', 'user-oleksandr', TIMESTAMP'2026-08-05 19:00:00', 412.0, 'UAH', 'expense', 'Аптека', 'Продукти', NULL, 'manual'),
('manual-0008', 'user-oleksandr', TIMESTAMP'2026-08-04 01:00:00', 15000.0, 'UAH', 'income', 'Клієнт - проєкт', 'Зарплата', NULL, 'manual'),
('manual-0009', 'user-oleksandr', TIMESTAMP'2026-07-12 19:00:00', 426.0, 'UAH', 'expense', 'Кафе', 'Фріланс', 'готівкою', 'manual'),
('manual-0010', 'user-oleksandr', TIMESTAMP'2026-07-15 00:00:00', 1124.0, 'UAH', 'expense', 'Аптека', 'Одяг', NULL, 'manual');
