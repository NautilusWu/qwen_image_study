FROM nvcr.io/nvidia/pytorch:26.07-py3

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/qwen-image

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["/bin/bash"]
