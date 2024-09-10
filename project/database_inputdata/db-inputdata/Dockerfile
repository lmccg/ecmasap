# Set base image
FROM python:3.10.4

# Set timezone
ENV TZ="Europe/Lisbon"

# Set Python Environment variable
ENV PYTHONUNBUFFERED=1

# Copy requirements.txt to /app/ (docker image)
COPY requirements.txt /app/

# Set working directory
WORKDIR /app

# Install linux libs & requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    freetds-dev \
    unixodbc-dev \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Import the Microsoft repository GPG keys
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -

# Download and install the Microsoft SQL Server package repository configuration
RUN curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Update the package index files
RUN apt-get update

# Install the ODBC Driver 17 for SQL Server package
RUN ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
    msodbcsql17

# Clean up package cache
RUN apt-get clean -y

# Set environment variable for SSL/TLS version
ENV SSL_VERSION="TLSv1.0"

# Update OpenSSL configuration
RUN sed -i '/\[system_default_sect\]/a MinProtocol = TLSv1.0\nCipherString = DEFAULT@SECLEVEL=1' /etc/ssl/openssl.cnf

# Manual installation of OpenSSL
RUN wget https://www.openssl.org/source/openssl-1.1.1l.tar.gz -O openssl.tar.gz \
    && tar -zxvf openssl.tar.gz \
    && cd openssl-* \
    && ./config \
    && make \
    && make install \
    && openssl version \
    && ldconfig \
    && cd .. \
    && rm -rf openssl.tar.gz openssl-*


# Copy /project to /app (docker image)
COPY /project .

# Volume
VOLUME /app

# EXPOSE port
EXPOSE 8080
