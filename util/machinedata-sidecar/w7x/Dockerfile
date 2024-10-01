FROM ubuntu

# Install dependencies
RUN apt-get update -y
RUN apt-get install -y gnupg

# Download encrypted files
WORKDIR /
RUN mkdir encrypted
ADD https://fz-juelich.sciebo.de/s/sGN7xVkkmEbCO7Y/download encrypted/w7xdb.sqlite.gpg

# Add decryption script
ADD decrypt.sh /
RUN chmod +x /decrypt.sh

# Set command
ENTRYPOINT /decrypt.sh