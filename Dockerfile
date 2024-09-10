FROM python:3.10.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    pkg-config \
    libatlas-base-dev \
    gfortran \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install --upgrade numpy
RUN pip install --upgrade tensorflow tensorflow-hub

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 5000

# Run the application with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]