# # Use the official Python image from the Docker Hub
# FROM python:3.8-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app

# # Copy the rest of the application code
# COPY . .

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Expose the port that Streamlit will run on
# EXPOSE 8501

# # Run Streamlit when the container launches
# CMD ["streamlit", "run", "app.py"]

# # Updated Docker file with copy ...

# FROM python:3.8-slim-buster
################################################################################################

# FROM python:3.9-slim-bullseye

# EXPOSE 8501

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libsqlite3-0 \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# COPY . /app

# RUN pip3 install -r requirements.txt

# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

################################################################################################


FROM python:3.9-slim-bullseye

EXPOSE 8501

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*


# Download and install SQLite from source
RUN wget https://www.sqlite.org/2023/sqlite-autoconf-3370200.tar.gz \
    && tar xvfz sqlite-autoconf-3370200.tar.gz \
    && cd sqlite-autoconf-3370200 \
    && ./configure --prefix=/usr --disable-static CFLAGS="-g -O2 -DSQLITE_ENABLE_FTS3=1 -DSQLITE_ENABLE_COLUMN_METADATA=1 -DSQLITE_ENABLE_UNLOCK_NOTIFY=1 -DSQLITE_SECURE_DELETE=1 -DSQLITE_ENABLE_DBSTAT_VTAB=1" \
    && make \
    && make install \
    && cd .. \
    && rm -rf sqlite-autoconf-3370200 sqlite-autoconf-3370200.tar.gz

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]