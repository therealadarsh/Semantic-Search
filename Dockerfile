FROM python:3.7

WORKDIR /application

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update  && apt-get install -y \
  python3-dev \
  apt-utils \
  python-dev \
  build-essential \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
COPY ./requirements.txt /application/requirements.txt

COPY ./data /application/data
COPY ./scripts /application/scripts

RUN pip install --no-cache-dir --upgrade -r /application/requirements.txt
RUN python /application/scripts/fetch_load_data.py
RUN python /application/scripts/embed_data.py --dataset /application/data/train[:12000] --split train

ENV PORT = 8000
EXPOSE 8000

CMD ["python", "/application/scripts/app.py"]