FROM python:3.10-slim

WORKDIR /outputs

RUN apt update
RUN apt-get install -y gcc g++

RUN pip install --upgrade pip
RUN pip install Cython

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt


