 
-- список таблиц
SELECT table_name FROM information_schema.tables
where table_schema = 'shipping';

-- список полей в конкретной таблице
SELECT *
FROM information_schema.columns
WHERE table_schema = 'shipping'
   AND table_name   = 'truck'
