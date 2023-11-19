FROM ic-registry.epfl.ch/mlo/pytorch:latest
LABEL authors="sallinen"

RUN apt-get update && apt-get install -y unzip  && rm -rf /var/lib/apt/lists/*

# Add your custom script
COPY install.sh /install_project.sh
COPY entrypoint.sh /entrypoint_new.sh
# Make your script executable
RUN chmod +x /install_project.sh


ENTRYPOINT ["/entrypoint_new.sh"]