FROM python:3.7-alpine
RUN apk add g++ freetype-dev sqlite
COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r ./requirements.txt
WORKDIR /home/backpropagation_py