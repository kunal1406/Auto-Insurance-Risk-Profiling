# Auto-Insurance-Risk-Profiling

clone the repository

MLFLOW_TRACKING_URI=https://dagshub.com/kunal1406/Auto-Insurance-Risk-Profiling.mlflow
MLFLOW_TRACKING_USERNAME=kunal1406
MLFLOW_TRACKING_PASSWORD=c1f8c1d6722f50e4980aec7e9eba0c1df1353ad6


$env:MLFLOW_TRACKING_URI = "https://dagshub.com/kunal1406/Auto-Insurance-Risk-Profiling.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "kunal1406"
$env:MLFLOW_TRACKING_PASSWORD = "c1f8c1d6722f50e4980aec7e9eba0c1df1353ad6"

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess


ECR repo link to store docker image
193501411007.dkr.ecr.us-east-2.amazonaws.com/autoinsurance

EC2 Instance

Open EC2 and Install docker in EC2 Machine:

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

# Configure EC2 as self-hosted runner

github setting>actions>runner>new self hosted runner> choose os> then run command one by one

# Setup github secrets

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = autoinsurance
