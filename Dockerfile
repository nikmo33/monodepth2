FROM eu.gcr.io/wayve-cloud/training_base:0.2.5

WORKDIR /app/nikhil

# pull the monodepth
RUN git clone https://github.com/nikmo33/monodepth2.git


