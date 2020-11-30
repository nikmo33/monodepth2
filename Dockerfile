FROM eu.gcr.io/wayve-cloud/training_base:0.2.5

WORKDIR /app/nikhil/

# pull the monodepth
RUN git clone https://github.com/nikmo33/monodepth2.git
RUN apt update && apt install -y \
    python3 \
    python3-pip && \
    pip3 install \
        numpy \
        pyyaml && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf ~/.cache/pip

WORKDIR /app/nikhil/monodepth2

RUN pip3 install -r requirements.txt
