## Basic image with scikit-learn and pandas
FROM python:3.12.6-slim

# Set working directory
WORKDIR /app

# Install dependencies: Azure SDKs, pandas, scikit-learn, joblib for saving model
RUN pip install --no-cache-dir \
    azure-identity \
    azure-storage-file-datalake \
    pandas \
    scikit-learn \
    joblib

# Copy your training script into the container
COPY train_model.py /app/train_model.py

# Run the training script by default
CMD ["python", "train_model.py"]