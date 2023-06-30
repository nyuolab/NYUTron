FROM nvcr.io/nvidia/tritonserver:22.02-py3 
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg
RUN /usr/bin/python3.8 -m pip install --upgrade pip
RUN pip3.8 install Cython
RUN pip3.8 install git+https://github.com/NVIDIA/NeMo.git@r1.6.2#egg=nemo_toolkit[nlp]