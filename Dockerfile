FROM python:3.7-alpine
WORKDIR /home/backpropagation_py
RUN apk add g++ freetype-dev
RUN pip install -r requirements.txt