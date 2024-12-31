# FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp4build
RUN mkdir -p ${STAGE_DIR}

##############################################################################
# Installation Basic Utilities
##############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common build-essential autotools-dev \
    nfs-common pdsh \
    cmake g++ gcc llvm-dev\
    curl wget vim tmux less unzip \
    htop iftop iotop ca-certificates openssh-server openssh-client \
    rsync iputils-ping net-tools\
    sudo

##############################################################################
# Installation Latest Git 
##############################################################################

# ENV http_proxy=http://127.0.0.1:8081
# ENV https_proxy=http://127.0.0.1:8081

RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git && \
    git --version

##############################################################################
# Client Liveness & Uncomment Port 22 for SSH Daemon
##############################################################################
# Keep SSH client alive from server side
RUN echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config
RUN cp /etc/ssh/sshd_config ${STAGE_DIR}/sshd_config && \
        sed "0,/^#Port 22/s//Port 22/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config


# ##############################################################################
# # Mellanox OFED
# ##############################################################################
# ENV MLNX_OFED_VERSION=5.4-1.0.3.0
# RUN apt-get install -y libnuma-dev
# RUN cd ${STAGE_DIR} && \
#         wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-${MLNX_OFED_VERSION}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64.tgz | tar xzf - && \
#         cd MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64 && \
#         ./mlnxofedinstall --user-space-only --without-fw-update --all -q && \
#         cd ${STAGE_DIR} && \
#         rm -rf ${STAGE_DIR}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu20.04-x86_64*

# ##############################################################################
# # nv_peer_mem
# ##############################################################################
# ENV NV_PEER_MEM_VERSION=1.1
# ENV NV_PEER_MEM_TAG=${NV_PEER_MEM_VERSION}-0
# RUN mkdir -p ${STAGE_DIR} && \
#         git clone https://github.com/Mellanox/nv_peer_memory.git --branch ${NV_PEER_MEM_TAG} ${STAGE_DIR}/nv_peer_memory && \
#         cd ${STAGE_DIR}/nv_peer_memory && \
#         ./build_module.sh && \
#         cd ${STAGE_DIR} && \
#         tar xzf ${STAGE_DIR}/nvidia-peer-memory_${NV_PEER_MEM_VERSION}.orig.tar.gz && \
#         cd ${STAGE_DIR}/nvidia-peer-memory-${NV_PEER_MEM_VERSION} && \
#         apt-get update && \
#         apt-get install -y dkms && \
#         dpkg-buildpackage -us -uc && \
#         dpkg -i ${STAGE_DIR}/nvidia-peer-memory_${NV_PEER_MEM_TAG}_all.deb

##############################################################################
# OPENMPI
##############################################################################
ENV OPENMPI_BASEVERSION=4.1
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.6

RUN cd ${STAGE_DIR} && \
        wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
        cd openmpi-${OPENMPI_VERSION} && \
        ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
        make -j"$(nproc)" install && \
        ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
        # Sanity check:
        test -f /usr/local/mpi/bin/mpic++ && \
        cd ${STAGE_DIR} && \
        rm -r ${STAGE_DIR}/openmpi-${OPENMPI_VERSION}
ENV PATH=/usr/local/mpi/bin:${PATH} \
        LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}
# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
        echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
        echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
        chmod a+x /usr/local/mpi/bin/mpirun

##############################################################################
# Python
##############################################################################
ENV PYTHON_VERSION=3

RUN add-apt-repository ppa:deadsnakes/ppa && \
        apt update && \
        apt install -y python3.12 python3.12-dev && \
        rm -f /usr/bin/python && \
        ln -s /usr/bin/python3.12 /usr/bin/python && \
        curl -O https://bootstrap.pypa.io/pip/3.7/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py && \
        pip install --upgrade pip && \
        # Print python an pip version
        python -V && pip -V

RUN rm -rf /usr/bin/python3 && \
        ln -s /usr/bin/python3.12 /usr/bin/python3 && \
        ln -s /usr/bin/python3.12-config /usr/bin/python3-config

##############################################################################
# Some Packages
##############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile-dev \
    libcupti-dev \
    libjpeg-dev \
    libpng-dev \
    screen \
    libaio-dev

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade six
RUN pip install numpy \
        scipy \
        scikit-learn \
        pandas \
        ipython \
        matplotlib \
        tqdm \
        mpi4py \
        wandb \
        pytest \
        pybind11

##############################################################################
# PyYAML build issue
# https://stackoverflow.com/a/53926898
##############################################################################
RUN rm -rf /usr/lib/python3/dist-packages/yaml && \
        rm -rf /usr/lib/python3/dist-packages/PyYAML-* && \
        pip install pyyaml

RUN pip install psutil \
        yappi \
        cffi \
        ipdb \
        pandas \
        matplotlib \
        py3nvml \
        pyarrow \
        graphviz \
        astor \
        boto3 \
        sentencepiece \
        msgpack \
        requests \
        sphinx \
        sphinx_rtd_theme \
        nvidia-ml-py3 \
        jieba \
        rouge-score \
        nltk \
        rouge \
        tabulate \
        fuzzywuzzy \
        mosaicml-streaming

RUN pip install aiohttp \
        aiosignal \
        annotated-types \
        anthropic \
        anyio \
        attrs \
        charset-normalizer \
        cohere \
        dataclasses-json \
        distro \
        filelock \
        frozenlist \
        h11 \
        httpcore \
        httpx \
        huggingface-hub \
        idna \
        jsonargparse \
        jsonpatch \
        jsonpointer \
        langchain \
        langchain-community \
        langchain-core \
        langsmith \
        langchain_openai \
        langchain_anthropic \
        langchain_cohere \
        marshmallow \
        multidict \
        mypy-extensions \
        numpy \
        openai \
        packaging \
        pydantic \
        pydantic_core \
        python-dotenv \
        regex \
        requests \
        sniffio \
        SQLAlchemy \
        tenacity \
        tiktoken \
        typing-inspect \
        typing_extensions \
        urllib3 \
        yarl

RUN python -c "import nltk; nltk.download('punkt_tab')"

##############################################################################
# Installation PyTorch
##############################################################################
ENV PYTORCH_VERSION=2.5.1

RUN pip config unset global.index-url
RUN pip install torch==${PYTORCH_VERSION} \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

RUN python -c "import torch; print(torch.__version__)"
                # print(torch.cuda.is_available()); \ # 在--gpus all之前False
                # a = torch.randn(10,10).to('cuda:0'); print(a+a)"

##############################################################################
# Installation Transformers
##############################################################################
ENV TRANSFORMERS_VERSION=4.46.1

RUN pip install transformers==${TRANSFORMERS_VERSION} \
        tokenizers \
        evaluate \
        datasets \
        accelerate
RUN python -c "import transformers; print(transformers.__version__)"

##############################################################################
# Installation Flash Attention
##############################################################################
RUN MAX_JOBS=128 pip install wheel && pip install --no-build-isolation flash-attn
RUN python -c "import flash_attn; print(flash_attn.__version__)"

##############################################################################
# DeepSpeed
##############################################################################
RUN git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed
RUN cd ${STAGE_DIR}/DeepSpeed && \
        git checkout . && \
        git checkout master && \
        rm deepspeed/ops/csrc && \
        rm deepspeed/ops/op_builder && \
        rm deepspeed/accelerator && \
        cp -R csrc op_builder deepspeed/ops/ && \
        cp -R accelerator deepspeed/ && \
        ./install.sh --allow_sudo
RUN rm -rf ${STAGE_DIR}/DeepSpeed
RUN python -c "import deepspeed; print(deepspeed.__version__)"

##############################################################################
# Installation VLLM
##############################################################################
ENV LD_LIBRARY=/usr/lib/

RUN apt install -y ccache

ENV VLLM_VERSION=0.6.3  

# RUN apt-get install -y nvidia-container-toolkit
# RUN nvidia-smi
# RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST=Ada
ENV DOCKER_BUILDKIT=0
# RUN python -c "import torch; print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_name(0))"
RUN git clone https://github.com/vllm-project/vllm.git -b v${VLLM_VERSION} ${STAGE_DIR}/vllm \
        && cd ${STAGE_DIR}/vllm \
        && python use_existing_torch.py \
        && pip install -r requirements-build.txt \
        && MAX_JOBS=8 pip install . --no-build-isolation
RUN rm -rf ${STAGE_DIR}/vllm
RUN python -c "import vllm; print(vllm.__version__)"

##############################################################################
# User
##############################################################################


##############################################################################
## Add deepspeed user
###############################################################################
# # Add a deepspeed user with user id 8877
# #RUN useradd --create-home --uid 8877 deepspeed
# RUN useradd --create-home --uid 1000 --shell /bin/bash deepspeed
# RUN usermod -aG sudo deepspeed
# RUN echo "deepspeed ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# # Change to non-root privilege
# USER deepspeed

# RUN useradd -m -s /bin/bash dev && \
#     echo "dev:password" | chpasswd

# RUN usermod -aG sudo dev && \
#     echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# USER dev

# WORKDIR /home/dev


##############################################################################
## SSH daemon port inside container cannot conflict with host OS port
###############################################################################
ENV SSH_PORT=7687
RUN cat /etc/ssh/sshd_config > ${STAGE_DIR}/sshd_config && \
        sed "0,/^Port 22/s//Port ${SSH_PORT}/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config
RUN cat /etc/ssh/ssh_config > ${STAGE_DIR}/ssh_config && \
        sed "0,/^#   Port 22/s//Port ${SSH_PORT}/" ${STAGE_DIR}/ssh_config > /etc/ssh/ssh_config
RUN echo "PermitRootLogin yes\n" >> /etc/ssh/sshd_config && \
        echo " StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
        echo " UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config
# RUN mkdir /dev/.ssh && ssh-keygen -t rsa -f ~/.ssh/id_rsa -P '' && cat /dev/.ssh/id_rsa.pub >> /dev/.ssh/authorized_keys
RUN mkdir /root/.ssh && ssh-keygen -t rsa -f ~/.ssh/id_rsa -P '' && cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys

RUN echo 'Host github.com\n\
\tHostname ssh.github.com\n\
\tPort 22\n\
\tUser git\n'\
> ~/.ssh/config



# CMD ["/bin/bash"]

# docker run -itd --name --ipc=host --net=host --privileged --gpus all -p 8081:8081 -v "D:\codes\sml:/mnt/sml" --name sml customized:v0.1