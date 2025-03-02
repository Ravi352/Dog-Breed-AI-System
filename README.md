# Dog-Breed-AI-System

The system follows some of the steps 

1. Cleaning the preprocessing the Dataframe which is analyzed and completed using some assumtions which is given in the "PreProcessing.ipynb" file.

As Data is having two types of tasks NLP as well as Data Analysis task, Some of the important steps are as follows

1. Langchain tool creation for handling the description related or NLP related questions.
2. For Data Analysis Tasks, pandas create agent are used which uses the dataframe and give the stats related answers.

For Choosing LLM, Open Source model as lamma 3.2 version is used. 
All the conversions and the query is saved in the conversational Memory with the user_id.

To Run the application proceed to the "OPKEY ASSIGNMENT" folder and install the requirements using "pip install -r requirements.txt" 
After installing all the neccesary requirements you can run the application using "streamlit run app.py" command.

For deployment to the production using docker, build docker image and push it to the production server.

Pull the image in the working environment and run the application.

