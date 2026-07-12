FROM public.ecr.aws/lambda/python:3.12

ARG TEXT_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TEXT_MODEL_ID=${TEXT_MODEL_ID} \
    TEXT_MODEL_PATH=/opt/models/text/all-MiniLM-L6-v2 \
    IMAGE_MODEL_PATH=/opt/models/torchvision/resnet50-imagenet1k-v2.pth \
    HF_HOME=/tmp/huggingface \
    TORCH_HOME=/tmp/torch \
    TOKENIZERS_PARALLELISM=false \
    HOME=/tmp

RUN dnf install -y libgomp \
    && dnf clean all \
    && rm -rf /var/cache/dnf

COPY requirements-torch.txt requirements.txt ${LAMBDA_TASK_ROOT}/

RUN python -m pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-torch.txt \
    && python -m pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt \
    && python -m pip check

COPY src ${LAMBDA_TASK_ROOT}/src
COPY scripts ${LAMBDA_TASK_ROOT}/scripts
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/lambda_handler.py

RUN cd ${LAMBDA_TASK_ROOT} \
    && python -m scripts.download_models \
    && python -m scripts.smoke_models \
    && rm -rf /root/.cache /tmp/huggingface /tmp/torch

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_MODEL_OFFLINE=1

CMD ["lambda_handler.handler"]
