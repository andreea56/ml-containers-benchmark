\# ML Containers Benchmark



Code and Kubernetes manifests for the paper "A Comparative Performance

Analysis of Supervised, Unsupervised, and Reinforcement Learning

Algorithms in Containerized Cloud Environments Using Docker and Kubernetes".



\## Dataset

Uses the "Fake and Real News Dataset" from Kaggle: \[link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset]



\## Running locally

pip install -r requirements.txt

python src/supervised/svm\_model.py



\## Running on Kubernetes

See kubernetes/ for manifests (Minikube and Azure AKS variants).



\## Results

Aggregated results and plots are in results/.

