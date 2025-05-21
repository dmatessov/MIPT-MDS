#!/bin/bash

docker build -f nginxDockerfile -t nginx-customized .

docker build -f pgDockerfile -t postgres-customized .

# docker run -p 8080:80 nginx-customized
