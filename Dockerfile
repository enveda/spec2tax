FROM python:3.8-slim
COPY ./src /code
#COPY .params /params
WORKDIR /code

RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get install -y zip unzip curl git
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install
RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install -r requirements.txt