# AWS Machine Learning Engineer Nanodegree program (Udacity)

This repository is the collection of projects that are part of the nanodegree program requirement.
The nanodegree program is 3-month self-paced certification program which was taken between 12/2022 - 03/2023. The program consists of 4 chapters and 1 capstone project. Each chapter consists of concept videos and coding exercises and quizzes and a trying-all-together project.

____

## [1. Prediction of Bike demand using AutoGluon](https://github.com/uyangas/AWS_MLE/tree/main/1_Predict_Bike_Sharing_Demand)

### Project goal:

To predict [bike demand](https://www.kaggle.com/c/bike-sharing-demand) using automl model `AutoGluon`. The section includes data exploration and hyperparameter tuning.

### Deliverables:
1. Project report ([`Bike Sharing Demand Prediction.md`](https://github.com/uyangas/AWS_MLE/blob/main/1_Predict_Bike_Sharing_Demand/Bike%20Sharing%20Demand%20Prediction.md))
1. Code notebook ([`Prediction Using AutoGluon.ipynb`](https://github.com/uyangas/AWS_MLE/blob/main/1_Predict_Bike_Sharing_Demand/Prediction%20Using%20AutoGluon.ipynb))
1. Code notebook's html ([`Prediction Using AutoGluon.html`](https://github.com/uyangas/AWS_MLE/blob/main/1_Predict_Bike_Sharing_Demand/Prediction%20Using%20AutoGluon.html))

____
## [2. Building ML Workflow on Amazon Sagamaker](https://github.com/uyangas/AWS_MLE/tree/main/2_ML_Workflow_on_Amazon_Sagemaker)

### Project goal:

To predict motorcycle and bike image from [`CIFAR-100`](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) dataset using Amazon's `image-classication-model`. The goal is to practice:
1. Load and upload data from and to S3
1. Train an estimator
1. Deploy the model through Endpoint
1. Setup Lambda function
1. Create Step function to demonstrate serverless workflow

### Deliverables:
1. Code notebook ([`Project2-ML_Workflow.ipynb`](https://github.com/uyangas/AWS_MLE/blob/main/2_ML_Workflow_on_Amazon_Sagemaker/Project2_ML_Workflow.ipynb))
1. Code notebook's html ([`Project2-ML_Workflow.html`](https://github.com/uyangas/AWS_MLE/blob/main/2_ML_Workflow_on_Amazon_Sagemaker/Project2-ML_Workflow.html))
1. Lambda function [scripts](https://github.com/uyangas/AWS_MLE/tree/main/2_ML_Workflow_on_Amazon_Sagemaker/Lambda%20Functions)
1. Screenshots of Step function workflow

____
## [3. Image Classification using AWS Sagemaker](https://github.com/uyangas/AWS_MLE/tree/main/3_Image_Classification_Using_AWS%20Sagemaker)

### Project goal:

To predict dog bread from image dataset using pretrained model through Sagemaker's script mode. The goal is to practice:
1. Create scripts to train and debug the model
1. Learn to use pretrained models
1. Hyperparameter tuning
1. Use Debugger and profiler
1. Deploy the best model
1. Package the model as docker container

### Deliverables:
1. Code notebook ([`train_and_deploy.ipynb`](https://github.com/uyangas/AWS_MLE/blob/main/3_Image_Classification_Using_AWS%20Sagemaker/train_and_deploy.ipynb))
1. Model training python script ([`train_model.py`](https://github.com/uyangas/AWS_MLE/blob/main/3_Image_Classification_Using_AWS%20Sagemaker/train_model.py))
1. Hyperparamter tuning python script ([`hpo.py`](https://github.com/uyangas/AWS_MLE/blob/main/3_Image_Classification_Using_AWS%20Sagemaker/hpo.py))
1. [README.md](https://github.com/uyangas/AWS_MLE/blob/main/3_Image_Classification_Using_AWS%20Sagemaker/README.md) file that explains the process
1. Screenshots of successfully trained hyperparameter tuning and model training processes

____
## [4. Operationalizing an AWS ML Project](https://github.com/uyangas/AWS_MLE/tree/main/4_Operationalizing%20an%20AWS%20ML)

### Project goal: 

To understand the operational aspects of model deployment including cost, efficiency, security and latency.
1. Compute resources to ensure efficient utilization
1. Distributed training
1. Learn to train model on EC2 instance
1. Construct pipelines for high throughput, low latency models
1. Design secure machine learning projects

### Deliverables:
1. Model training notebook containing hyperparameter tuning, model training, distributed training and model deployment ([`train_and_deploy-solution.ipynb`](https://github.com/uyangas/AWS_MLE/blob/main/4_Operationalizing%20an%20AWS%20ML/train_and_deploy-solution.ipynb))
1. Model train and hyperparameter tuning script ([`hpo.py`](https://github.com/uyangas/AWS_MLE/blob/main/4_Operationalizing%20an%20AWS%20ML/hpo.py))
1. Model training script on EC2 instance ([`ec2train1.py`](https://github.com/uyangas/AWS_MLE/blob/main/4_Operationalizing%20an%20AWS%20ML/ec2train1.py))
1. Lambda function script ([`lambdafunction.py`](https://github.com/uyangas/AWS_MLE/blob/main/4_Operationalizing%20an%20AWS%20ML/lamdafunction.py))
1. PDF file that explains the process ([`writeup.pdf`](https://github.com/uyangas/AWS_MLE/blob/main/4_Operationalizing%20an%20AWS%20ML/writeup.pdf))
1. [Screenshots](https://github.com/uyangas/AWS_MLE/tree/main/4_Operationalizing%20an%20AWS%20ML/Screenshots) of project steps

____
## [5. Capstone project]()

### Project goal:

### Deliverables:

____