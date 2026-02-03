# End-to-End MLOps Pipeline on Azure – Churn Prediction

This project demonstrates an end-to-end MLOps workflow for deploying a machine learning model on Azure using Docker and Kubernetes.

## 🔹 Project Overview
The goal of this project is to predict customer churn using a machine learning model and deploy it as a scalable REST API.

## 🔹 Tech Stack
- Python
- scikit-learn
- FastAPI
- Docker
- Azure Container Registry (ACR)
- Azure Kubernetes Service (AKS)

## 🔹 Architecture

## 🔹 Project Structure
- `src/` – Training, preprocessing, and evaluation scripts
- `model/` – Saved model and preprocessing pipeline
- `docker/` – Dockerfile and FastAPI inference app
- `pipelines/` – MLOps pipeline components
- `requirements.txt` – Python dependencies

## 🔹 Model Training
- Data preprocessing using pandas and scikit-learn
- Categorical encoding and feature scaling
- Model training and evaluation
- Model and preprocessing pipeline saved using joblib

## 🔹 Deployment
- FastAPI used to expose the model as a REST API
- Docker used to containerize the inference service
- Image pushed to Azure Container Registry
- Deployed to Azure Kubernetes Service for scalable inference

## 🔹 Key Learnings
- Containerizing ML models using Docker
- Deploying ML services on Kubernetes
- Handling real-world MLOps deployment challenges
- Ensuring consistent preprocessing between training and inference

## 🔹 Future Improvements
- Add CI/CD using Azure DevOps
- Integrate Azure ML for experiment tracking
- Add monitoring and logging

---
