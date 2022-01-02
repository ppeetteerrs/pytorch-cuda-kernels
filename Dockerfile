FROM nvcr.io/nvidia/pytorch:21.12-py3

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Change shell
SHELL ["/bin/bash", "-c"]

# Add User
RUN groupadd --gid $USER_GID $USERNAME \
	&& useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
	&& apt-get update -y \
	&& apt-get upgrade -y \
	&& apt-get install -y sudo git wget curl htop build-essential ninja-build \
	&& echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
	&& chmod 0440 /etc/sudoers.d/$USERNAME

# Install conda packages for VSCode
RUN conda install -y black flake8 autoflake

USER user

# Include CUDA binaries in PATH
ENV PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH

# Create local conda environment
RUN conda init --all \
	&& conda create --name user --clone base
RUN echo "conda activate user" >> /home/user/.bashrc

WORKDIR /home/user

CMD "bash"