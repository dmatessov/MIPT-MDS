-- Создаем пользователя
DO $$ 
BEGIN
  IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'test') THEN
    CREATE USER test WITH ENCRYPTED PASSWORD 'secret';
  END IF;
END $$;


-- Create the database if it doesn't exist
SELECT 'CREATE DATABASE test'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'test')\gexec
-- Grant privileges to the user on the database
GRANT ALL PRIVILEGES ON DATABASE test TO test;
