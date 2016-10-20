FROM python:3.5

RUN apt-get update && apt-get install -y gfortran liblapack-dev libblas-dev
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
  bash miniconda.sh -b -p $HOME/miniconda

COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp35-cp35m-linux_x86_64.whl

COPY ./src /src

ENV PATH $HOME/miniconda/bin:$PATH
ENV PYTHONHASHSEED 2157

ENTRYPOINT ["python", "src/main/python/server.py"]
