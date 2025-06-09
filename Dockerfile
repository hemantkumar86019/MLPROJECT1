FROM python:3.11-slim-buster

# Set working directory inside container
WORKDIR /app

# Copy everything into container
COPY . /app

# Install system tools (optional)
RUN apt update -y && apt install -y awscli

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Flask default port
EXPOSE 5000

# Run Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "application:application"]
