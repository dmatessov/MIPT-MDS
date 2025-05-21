# Используем официальный образ Alpine для Nginx
FROM nginx:alpine

# Копируем файл конфигурации Nginx с настройкой запрета POST-запросов
COPY nginx.conf /etc/nginx/nginx.conf

# Экспонируем порт 80 для HTTP-запросов
EXPOSE 80

# Запускаем Nginx в режиме форграунда
CMD ["nginx", "-g", "daemon off;"] 
