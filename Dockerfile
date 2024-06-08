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

FROM python:3.9-slim-bullseye

EXPOSE 8501

RUN apt-get update && apt-get install -y \
    build-essential \
    libsqlite3-0 \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

################################################################################################