import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
import langchain.globals as lcg  # Import langchain.globals

# Set verbose to True or False based on your requirements
lcg.set_verbose(True)  # Enable verbose mode if needed

# Set up the model and prompt template
os.environ["GOOGLE_API_KEY"] = 'AIzaSyAuoLc2fLBrNLyIoQX8aJkYmu9B8ymsxHg'
generation_config = {"temperature": 0.6, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
model = GoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

prompt_template_resto = PromptTemplate(
    input_variables=['age', 'gender', 'weight', 'height', 'mood', 'fitness_goals'],
    template="Diet Recommendation System:\n"
             "I want you to recommend 10 workout names, "
             "based on the following criteria:\n"
             "Person age: {age}\n"
             "Person gender: {gender}\n"
             "Person weight: {weight}\n"
             "Person height: {height}\n"
             "Person mood: {mood}\n"
             "Person fitness_goals: {fitness_goals}\n"
)
chain_resto = LLMChain(llm=model, prompt=prompt_template_resto)

# Custom styling
st.markdown(
    """
    <style>
        .title {
            font-size: 25px;
            font-weight: bold;
            font-family: 'Arial', sans-serif;
        }
        .content {
            font-family: 'Helvetica', sans-serif;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a Streamlit web app
st.title('StayFit Workout Recommendation')

# User input form
age = st.text_input('Age')
gender = st.selectbox('Gender', ['Male', 'Female'])
weight = st.text_input('Weight (kg)')
height = st.text_input('Height (cm)')
mood = st.selectbox('Mood', ['Happy', 'Sad', 'Neutral'])
fitness_goals = st.selectbox('Fitness Goals', ['Weight Loss', 'Weight Gain', 'Muscle Gain'])

# Button to trigger recommendations
if st.button('Get Recommendations'):
    # Check if all form fields are filled
    if age and gender and weight and height and mood and fitness_goals:
        input_data = {
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'mood': mood,
            'fitness_goals': fitness_goals
        }

        results = chain_resto.invoke(input_data)

        # Extract recommendations
        results_text = results['text']
        st.write("Generated Recommendations:")
        st.write(results_text)
    else:
        st.write("Sorry, you did not provide any information. Please fill in all the form fields.")
