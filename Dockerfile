FROM python:3.10.6-slim

WORKDIR /outputs

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt


