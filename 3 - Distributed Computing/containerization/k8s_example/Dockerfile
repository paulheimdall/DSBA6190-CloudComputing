## Basic image with scikit-learn and pandas
FROM python:3.12.6-slim

# Set working directory
RUN mkdir /lab 
WORKDIR /lab

# Install dependencies: Azure SDKs, pandas, scikit-learn, joblib for saving model
RUN python -m pip install --no-cache-dir \
    azure-identity \
    azure-storage-file-datalake \
    pandas \
    scikit-learn \
    joblib

# Copy your training script into the container
COPY train.py /lab/train.py
COPY BostonHousing.csv /lab/BostonHousing.csv

# Run the training script by default
CMD ["python", "train.py"]