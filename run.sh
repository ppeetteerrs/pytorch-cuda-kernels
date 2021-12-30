docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 8888:8888 -v $(pwd):/workspace -it nvcr.io/nvidia/pytorch:21.12-py3 bash