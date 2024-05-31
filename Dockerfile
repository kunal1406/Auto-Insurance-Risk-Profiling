# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the rest of the application code
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit will run on
EXPOSE 8501

# Run Streamlit when the container launches
CMD ["streamlit", "run", "app.py"]

# Updated Docker file with copy ...