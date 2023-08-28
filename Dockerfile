FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get install -y python3.8
RUN apt-get install -y python3-pip
RUN apt-get install -y python3.8-dev
RUN apt-get install -y python3.8-distutils

RUN apt-get install -y git

WORKDIR /app

COPY . .
RUN python3.8 -m pip install -r requirements.txt

CMD ["python3.8", "-m", "uvicorn", "--workers", "1", "--port", "8080", "--host", "0.0.0.0", "main:app"]

EXPOSE 8080


