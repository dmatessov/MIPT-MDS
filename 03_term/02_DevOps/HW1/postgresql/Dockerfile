# Используем официальный образ PostgreSQL
FROM postgres:latest

# Устанавливаем переменные окружения для создания пользователя и базы данных
ENV POSTGRES_USER test
ENV POSTGRES_PASSWORD secret
ENV POSTGRES_DB test

# Копируем скрипт инициализации в контейнер
COPY init.sql /docker-entrypoint-initdb.d/
