# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-04.html
FROM nvcr.io/nvidia/pytorch:22.04-py3

COPY ./requirements.txt /app/requirements.txt
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r /app/requirements.txt

COPY ./lib/pydock-0.4-cp38-cp38-manylinux_2_31_x86_64.whl /app/pydock-0.4-cp38-cp38-manylinux_2_31_x86_64.whl
RUN pip install /app/pydock-0.4-cp38-cp38-manylinux_2_31_x86_64.whl
