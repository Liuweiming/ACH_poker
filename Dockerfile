From nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN rm -rf /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    apt update && \
    apt install -y software-properties-common build-essential ssh git curl wget \
    python3-dev python3-pip python3-setuptools python3-wheel \
    autoconf automake libtool libffi-dev vim unzip zip zlib1g-dev liblzma-dev rsync && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 100

RUN wget https://mirrors.huaweicloud.com/bazel/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh && \
    chmod +x ./bazel-0.26.1-installer-linux-x86_64.sh && \
    ./bazel-0.26.1-installer-linux-x86_64.sh

RUN git clone --branch v1.15.3  https://github.com/tensorflow/tensorflow

RUN cd /tensorflow && ./configure && \
    bazel build --config=opt --config=cuda --config=monolithic  //tensorflow:libtensorflow_cc.so

RUN mkdir /third_party && cd /third_party && mkdir tensorflow && cd tensorflow && \
    mkdir include && \
    cp -rL /tensorflow/tensorflow ./include && \
    cp -rL /tensorflow/third_party ./include && \
    cp -rL /tensorflow/bazel-tensorflow/external/nsync ./include && \
    cp -rL /tensorflow/bazel-tensorflow/external/eigen_archive ./include && \
    cp -rL /tensorflow/bazel-tensorflow/external/com_google_absl ./include && \
    cp -rL /tensorflow/bazel-tensorflow/external/com_google_protobuf ./include && \
    mkdir bazel_include && \
    rsync -avm --include='*.h'  -f 'hide,! */' /tensorflow/bazel-bin/ ./bazel_include && \
    mkdir bin && \
    cp -r /tensorflow/bazel-bin/tensorflow/libtensorflow_cc* ./bin

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN rm -rf /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    apt update && \
    apt install -y software-properties-common build-essential ssh git curl wget \
    python3 python3-dev python3-pip python3-setuptools python3-wheel vim \ 
    wget autoconf automake libtool libffi-dev vim unzip zip zlib1g-dev liblzma-dev rsync && \
    wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt update && apt install -y cmake && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 100

RUN wget -qO - http://download.savannah.gnu.org/releases/libunwind/libunwind-1.1.tar.gz | tar -xz && \
    cd libunwind-1.1 && ./configure && make -j8 && make install && \
    cd ../ && rm -rf libunwind-1.1 && \
    wget -qO - https://github.com/gperftools/gperftools/releases/download/gperftools-2.8/gperftools-2.8.tar.gz | tar -xz && \
    cd gperftools-2.8 && ./configure && make -j8 && make install && \
    cd ../ && rm -rf gperftools-2.8 && \
    ldconfig


RUN mkdir my_spiel
WORKDIR /my_spiel

COPY . .
COPY --from=0 /third_party ./third_party

ENV CC=gcc
ENV CXX=g++
ENV BUILD_WITH_ACPC=ON
ENV PYTHONPATH=${PYTHONPATH}:/my_spiel/third_party/open_spiel
ENV PYTHONPATH=${PYTHONPATH}:/my_spiel/build/third_party/open_spiel/python


RUN cd third_party && \
    ./install.sh 
# pip install --upgrade -r requirements.txt && \
# cd .. && \
# mkdir build && cd build && \
# cmake -DCMAKE_BUILD_TYPE=Release .. && \ 
# make -j8