# FROM (base_image:tag)
# tensorflow images are based on ubuntu16.04 image
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

ENV http_proxy http://proxy-chain.intel.com:911
ENV https_proxy http://proxy-chain.intel.com:911

COPY apt.conf /etc/apt
COPY pip.conf /etc

# base gosu related script
RUN apt-get update
RUN apt-get -y --no-install-recommends install \
    ca-certificates \
    curl

RUN gpg --keyserver ha.pool.sks-keyservers.net --keyserver-options http-proxy=http://proxy-chain.intel.com:911 --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4

RUN curl -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture)" \
    && curl -o /usr/local/bin/gosu.asc -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture).asc" \
    && gpg --verify /usr/local/bin/gosu.asc \
    && rm /usr/local/bin/gosu.asc \
    && chmod +x /usr/local/bin/gosu

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# main script
RUN apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libcap-dev \
    python3-pip \
    python3-dev

RUN apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    libgtk-3-0 \
    libgtk-3-dev \
    lsb-release

RUN cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip

RUN pip3 --no-cache-dir install Cython numpy setuptools
RUN pip3 --no-cache-dir install torch torchvision
COPY requirements.txt /tmp
RUN pip3 --no-cache-dir install -r /tmp/requirements.txt

RUN echo "alias python=python3" >> /root/.bashrc

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

