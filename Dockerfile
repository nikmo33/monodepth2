FROM eu.gcr.io/wayve-cloud/training_base:0.2.5

WORKDIR /app/nikhil/


# pull the monodepth
RUN apt update && apt install -y \
    python3 \
    python3-pip
RUN pip3 install \
        numpy \
        pyyaml
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf ~/.cache/pip

# RUN pip3 install pytorch-lightning
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt



