FROM ic-registry.epfl.ch/mlo/pytorch:latest
MAINTAINER "Alexandre Sallinen  <alexandre.sallinen@epfl.ch>"
RUN apt-get update && apt-get install -y unzip
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update && apt-get install -y cuda-nvcc-11-7  && rm -rf /var/lib/apt/lists/*

# Add your custom script
COPY install.sh /install_project.sh
COPY entrypoint.sh /entrypoint_new.sh
# Make your script executable
RUN chmod +x /install_project.sh
RUN chmod +x /entrypoint_new.sh

ARG NB_USER=sallinen
ARG NB_UID=30217
ARG NB_GROUP=MLO-unit
ARG NB_GID=204162

ENTRYPOINT ["/entrypoint_new.sh"]