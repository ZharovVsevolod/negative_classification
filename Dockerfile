FROM python:3.10-slim

RUN apt update
RUN apt-get install -y gcc g++

RUN pip install --upgrade pip
RUN pip install Cython

WORKDIR /usr/app/src

COPY . .
RUN python3 -m pip install -r requirements.txt

# CMD ["python3", "-m" "/usr/app/src/negative_classification/train"]