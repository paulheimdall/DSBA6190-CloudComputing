## Basic image with scikit-learn and pandas
FROM python:3.12.6-slim

# Set working directory
RUN mkdir /lab 
WORKDIR /lab

# Install dependencies: pandas, scikit-learn, joblib for saving model
RUN python -m pip install pandas \
    scikit-learn \
    joblib

# Copy your training script and data  into the container
COPY train.py /lab/train.py

# Run the training script by default
CMD ["python", "train.py"]