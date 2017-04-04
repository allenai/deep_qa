FROM allenai-docker-private-docker.bintray.io/cuda:8

ENTRYPOINT ["/bin/bash", "-c"]

WORKDIR /stage

RUN conda create -n runenv --yes python=3.5
ENV PYTHONHASHSEED 2157

# Copying and installing required python packages.
COPY requirements.txt requirements.txt
# requirements.txt uses the CPU version of tensorflow, for Travis CI; we want the GPU version, so
# we use sed to change it before installing the requirements.
RUN sed -ie 's/^tensorflow/tensorflow-gpu/g' requirements.txt
RUN /bin/bash -c "source activate runenv && pip install -r requirements.txt"
RUN /bin/bash -c "source activate runenv && python -m nltk.downloader punkt"
RUN /bin/bash -c "source activate runenv && python -m spacy.en.download all"

# Copying the source code.
COPY scripts/ scripts/
COPY deep_qa/ deep_qa/

# Parameter file to run - this is an argument to the `docker build` command.
ARG PARAM_FILE
COPY $PARAM_FILE model_params.json

CMD ["source activate runenv && exec python scripts/run_model.py model_params.json"]
