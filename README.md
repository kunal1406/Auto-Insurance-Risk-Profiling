# Auto-Insurance-Risk-Profiling

### How to Run

Follow these steps to get the project up and running on your machine:

#### Steps:

**Clone the Repository**
- First, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/kunal1406/Auto-Insurance-Risk-Profiling.git
```

**STEP 01: Create a Conda Environment**
- After cloning the repository, navigate to the directory where the repository is cloned and create a Conda environment:
```bash
conda create -p venv  python==3.9 -y 
conda activate venv
```
**STEP 02: Install the Requirements**
- Install all the brequired packages from the 'requirements.txt' file:
```bash
pip install -r requirements.txt
```

**STEP 03: Execute the Main Script**
- Execute the main script to run the pipelines and generate intermediate files
```bash
python main.py
```

**STEP 04: Run the Application**
- Start the application by running
```bash
python app.py
```

# Yet to be updated
```bash
MLFLOW_TRACKING_URI=https://dagshub.com/kunal1406/Auto-Insurance-Risk-Profiling.mlflow
MLFLOW_TRACKING_USERNAME=kunal1406
MLFLOW_TRACKING_PASSWORD=c1f8c1d6722f50e4980aec7e9eba0c1df1353ad6
```
*Set the variables in the environment*

```bash
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/kunal1406/Auto-Insurance-Risk-Profiling.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "kunal1406"
$env:MLFLOW_TRACKING_PASSWORD = "c1f8c1d6722f50e4980aec7e9eba0c1df1353ad6"
```
# Policy:
```bash
1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess
```

# ECR repo link to store docker image
```bash
193501411007.dkr.ecr.us-east-2.amazonaws.com/autoinsurance
```
# EC2 Instance

Open EC2 and Install docker in EC2 Machine:
```bash
sudo apt-get update -y

sudo apt-get upgrade
```
#required
```bash
curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```
# Configure EC2 as self-hosted runner
```bash
github setting>actions>runner>new self hosted runner> choose os> then run command one by one
```
# Setup github secrets
```bash
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-2

AWS_ECR_LOGIN_URI = 193501411007.dkr.ecr.us-east-2.amazonaws.com


ECR_REPOSITORY_NAME = autoproj
```
# Link for the App
```bash
http://18.224.141.64:8501/

http://3.19.64.34:8501/
```
