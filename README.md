# Auto Insurance Risk Profiling

For a visual guide on how the tool works, please watch the following video

https://www.youtube.com/watch?v=D8Uy5b8NTp0

## Purpose

* This project develops an end to end tool designed to assist insurance managers and marketing teams in targeting potential customers, aiming to minimize claim amounts. 
* By leveraging advanced predictive analytics, the tool identifies customers with the lowest risk profiles- those least likely to file claims and those with minimal cost per claim when they occur. 
* The insights generated allow the creation of tailored marketing campaigns that effectively engage different customer segments based on their calculated risk levels.

## Functionality

* Classification Model: Used to determine the likelihood of a claim being filed by potential customers.
* Regression Model: Predict the possible cost of a claim should one occur.
* Score-Based Segmentation: Customers are segmented into low, medium, and high-risk profiles based on scores derived from predictive models.
* Statistical Analysis: This involves testing the significance of the identified groups to ensure they are statistically valid and not due to random chance. Post-hoc analysis is    employed for deeper verification.
* Retrieval-Augmented Generation (RAG): This system enhances non-technical user interaction with comprehensive risk profile reports, databases of risk profiles,  and logged statistical analysis results.
* User Interface: A Streamlit-based frontend application allows easy access and interaction with the analytical outputs and functionalities.

## Goals

1) Minimize Claim Amounts: optimize profitability and risk management
2) Enhance Decision Making: Auto manager and Marketing Team can take informed decisions tailored to customer risk profiles based on actionable insights 
3) Automate and Streamline Operations: Enhance user access through deployment

## Deployment

The tool is containerized using Docker for consistent deployment environments and pushed to an Elastic Container Repository. It incorporates IAM security measures to manage access securely. The Docker image is deployed on an EC2 instance to ensure scalability and reliability. Automation of updates and deployment is managed through GitHub Actions, establishing a CI/CD pipeline that enhances development efficiency and operational stability.

# Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)


## Features

1) **Risk Profile Prediction Report** <br>
    This feature generates comprehensive risk profiles for customers by analyzing provided data attributes. It offers tailored recommendations for the marketing team, highlighting key indicators and potential        customer insights. The report includes:
    
    Risk Assessment: A dynamic evaluation of the customer's risk based on historical and contextual data.
    Actionable Insights: Specific recommendations to guide marketing strategies, crafted from the risk assessment results.

2) **Risk Group Analysis Dashboard** <br>
    This feature delivers both broad overviews and in-depth analyses of customer risk groups, facilitating strategic decision-making:
    
    Overview: Displays a summary of risk groups, including the distribution of key attributes among these groups. This helps in quickly grasping the risk landscape across different segments.
    Detailed Analysis: Provides exhaustive analytics on the attributes of each risk group, with detailed distribution data and trends analysis.

    Additionally, this feature includes a chat interface that allows non-technical users, such as marketing team members or auto managers, to interact with the system. Users can ask questions in natural         language, which are translated into SQL queries to retrieve information from a database of processed files with predicted outcomes.

3) **Statistical Analysis of Risk Profiles** <br>
    This component offers detailed statistical analysis of customer profiles across three key dimensions:
    
    Demographic Analysis: Analyzes customer data by age, gender, and other demographic factors to understand risk patterns.
    Financial and Regional Analysis: Evaluates financial behaviors and regional characteristics to assess their impact on insurance risks.
    Vehicular Analysis: Studies vehicle-related factors to correlate with risk levels.

    For each section, the tool provides:
    
    Descriptive Statistics: Basic statistical measures that describe the data set.
    Visualizations: Graphs and charts that illustrate data distributions and trends.
    Hypothesis Testing Logs: Documentation of statistical tests used to verify the significance of observed relationships, ensuring they are not due to random variation. It also includes post-hoc analyses for        detailed insights.
    
    Similar to the Risk Group Analysis Dashboard, this feature integrates a chat function enabling non-technical users to discuss and interpret the statistical findings in simpler terms. This interaction helps       bridge the gap between complex data insights and actionable business intelligence.




## Installation

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




## Usage

For a visual guide on how the tool works, please watch the following video

https://www.youtube.com/watch?v=D8Uy5b8NTp0





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
