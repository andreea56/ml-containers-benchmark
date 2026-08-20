# ML Containers Benchmark

Source code, Docker images, and Kubernetes manifests for the paper:

"A Comparative Performance Analysis of Supervised, Unsupervised, and 
Reinforcement Learning Algorithms in Containerized Cloud Environments 
Using Docker and Kubernetes"

Andreea-Oana Radu, Sergiu-Alexandru Ionescu, Ioana Nagîț  
Bucharest University of Economic Studies, Romania

## Overview

This repository contains the complete experimental pipeline used to 
benchmark ten machine learning algorithms across three learning 
paradigms (supervised, unsupervised, reinforcement learning), deployed 
identically across four environments: a local Python virtual 
environment, Docker (without orchestration), Kubernetes on a local 
Minikube cluster, and Kubernetes on a cloud cluster managed by Azure 
Kubernetes Service (AKS).

## Algorithms

- Supervised: SVM, Random Forest, Logistic Regression, DeepText (CNN)

- Unsupervised: K-Means, DBSCAN
- Reinforcement learning: SARSA and DQN, each evaluated both in their
- native environments (FrozenLake, CartPole) and reformulated as
- single-step contextual bandits for direct comparison with the
- classification algorithms

## Dataset

This project uses the public "Fake and Real News Dataset" from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset itself is not included in this repository. Download it from
Kaggle and place `fake.csv` and `true.csv` in the `data/` folder before
running any script.

## Repository structure

ml-containers-benchmark/

├── README.md                          # Project description + how to run it                          
├── requirements.txt                   # Python dependencies
├── data/
│   └── README.md                      # link to the Kaggle dataset
├── src/
│   ├── common/
│   │   └── data_utils.py
│   ├── supervised/
│   │   ├── svm_model.py
│   │   ├── random_forest_model.py
│   │   ├── logistic_regression_model.py
│   │   └── deeptext_model.py
│   ├── unsupervised
│   │   ├── kmeans_model.py
│   │   └── dbscan_model.py
│   ├── reinforcement
│   │   ├── sarsa_agent.py
│   │   ├── dqn_agent.py
│   │   ├── sarsa_bandit_news.py
│   │   └── dqn_bandit_news.py
│   ├── aggregate_results.py
│   └── plot_results.py
├── docker/
│   ├── Dockerfile.supervised
│   ├── Dockerfile.unsupervised
│   ├── Dockerfile.reinforcement
│   └── Dockerfile.plots
├── kubernetes/
│   ├── 00-storage-azure.yaml
│   ├── 05-rbac-azure.yaml
│   ├── 10-supervised-jobs-azure.yaml
│   ├── 20-unsupervised-jobs-azure.yaml
│   ├── 30-reinforcement-jobs-azure.yaml
│   └── 40-plots-job-azure.yaml
└── results/
├── aggregated\_results.csv         # final, aggregated results
└── plots/                          # the 6 final charts

## Running locally
---bash

pip install -r requirements.txt

export DATA\_DIR=data
export OUTPUT\_DIR=output
export SEED=42

python src/supervised/svm\_model.py

## Running on Kubernetes

Manifests for both Minikube (local) and Azure Kubernetes Service (cloud)
are provided in `kubernetes/`. Each algorithm category is run as a
Kubernetes Indexed Job, with five parallel repetitions mapped to five
fixed seed values (42, 123, 2024, 7, 99).

---bash

kubectl apply -f kubernetes/00-storage-azure.yaml
kubectl apply -f kubernetes/05-rbac-azure.yaml
kubectl apply -f kubernetes/10-supervised-jobs-azure.yaml
kubectl apply -f kubernetes/20-unsupervised-jobs-azure.yaml
kubectl apply -f kubernetes/30-reinforcement-jobs-azure.yaml
kubectl apply -f kubernetes/40-plots-job-azure.yaml

## Results

Aggregated results (mean ± standard deviation across five seeds) and the
final comparison figures are available in `results/`.

## Citation

If you use this code, please cite:
Radu, A.-O., Ionescu, S.-A., Nagîț, I. (2026). A Comparative Performance Analysis of Supervised, Unsupervised, and Reinforcement Learning 
Algorithms in Containerized Cloud Environments Using Docker and Kubernetes. Journal name: DBJOURNAL

