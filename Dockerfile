FROM 896129387501.dkr.ecr.us-west-2.amazonaws.com/infrastructure/aristo/cuda:8

ENTRYPOINT ["/bin/bash", "-c"]

WORKDIR /stage

RUN conda create -n runenv --yes python=3.5
ENV PYTHONHASHSEED 2157

# Copy the pip requirements and the scripts.
COPY requirements.txt requirements.txt
COPY scripts/ scripts/

# requirements.txt uses the CPU version of tensorflow. We want the GPU version, so
# we change it before installing the requirements.
ARG PROCESSOR=gpu
RUN scripts/set_processor.sh $PROCESSOR

RUN /bin/bash -c "source activate runenv && scripts/install_requirements.sh"

# Copying the source code later, to minimize image rebuilds
COPY deep_qa/ deep_qa/

# Parameter file to run - this is an argument to the `docker build` command.
ARG PARAM_FILE
COPY $PARAM_FILE model_params.json

CMD ["source activate runenv && exec python scripts/run_model.py model_params.json"]
