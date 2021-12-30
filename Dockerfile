FROM nvcr.io/nvidia/pytorch:21.12-py3

RUN conda install -y black flake8 jupyter notebook

CMD "bash"