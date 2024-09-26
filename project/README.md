# MAS system

## This system:

### Manages the communication between agents
### Has agents designed for specific tasks
### Has agents such as:
#### MainAgent &rarr; responsible for all agents' communications, receives all requests and gives responses to the user. 
#### DbAgent &rarr; responsible for all database's operations , such as saving the ml model in database
#### EvaluatorAgent &rarr; responsible for evaluate the ml models' results 
#### HistoricalDataAgent &rarr; responsible for obtain the historical data from microservice and pré process all the data to be used in train/retrain/predict 
#### MLModelAgent &rarr; responsible for execute the ml models' operations, such as train, retrain or predict
#### PublisherAgent &rarr; responsible for publish all ml models' results
#### RLRetrainAgent &rarr; responsible for determine which models should be retrained and when
#### RLSelectorAgent &rarr; responsible for all determine which model(s) should be used in the prediction 
#### RulesAgent &rarr; responsible for create, update and provide rules to fixe the predictions, if needed
#### SubscriberAgent &rarr; responsible for check the topics and messages published about the results 
#### TargetAgent &rarr; responsible for a target operation. Retrieves the specif data needed for that target to be trained, retrained or predicted
#### XAIAgent &rarr; responsible for explain the ml models' decisions

### Has a micro service to retrieve historical data

## Production Environment folder

### To run this project locally (Windows):

#### First create a local instance of the database to be used:
##### 1. Install docker desktop
##### 2. Open docker desktop
##### 3. In a terminal run:
###### docker run -d --name db_local -e POSTGRES_PASSWORD=psw -p 9911:5432 postgres

#### Then create a virtual environment in the terminal running:

##### python -m venv venv

#### Then activate the virtual environment:

##### venv\Scripts\activate

#### Now, install all the libraries present in the requirements.txt file (Library versions are also in the file):

##### pip install -r .\requirements.txt

#### 1- In one terminal:
##### To run the microservice for historical data (It will run the service in port 8080, so url service = 127.0.0.1:8080):

###### hypercorn app:app -b 0.0.0.0:8080

#### 2- In another terminal:
##### To run the multi-agent system

###### Follow the tutorial in https://github.com/gecad-group/peak-mas/tree/main/docker to create the docker for the XMPP Server
###### Once done, go to the directory where the file run.sh is located.
###### Run: bash .\run.sh