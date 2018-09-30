FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER Lennart Van der Goten
LABEL Description="MT"
COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt
RUN python3 -m spacy download en
RUN python3 -m spacy download de

RUN mkdir -p /src
RUN mkdir -p /data
COPY ./*.py /src/
COPY config.yml /src/
ENV PYTHONPATH=/src

# Encoding
RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR /src
ENTRYPOINT ["python3", "-b", "orchestrate.py"]