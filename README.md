🔎 AI Fake News Detector with LIME Explanations
This is a web application built with Python and Streamlit to classify English news articles as Real or Fake. The project's key feature is its use of LIME (Local Interpretable Model-agnostic Explanations) to provide visual explanations, helping users understand the reasoning behind the AI's predictions.

✨ Key Features
News Classification: Utilizes a Scikit-learn PassiveAggressiveClassifier model to classify text.

AI Explainability (XAI): Integrates the LIME library to highlight the keywords that most influence the prediction outcome, enhancing model transparency.

Interactive UI: A user-friendly and interactive web interface built with Streamlit.

Train & Save: The model training pipeline is provided in a Jupyter Notebook (Model_Training.ipynb), and the trained objects (model and vectorizer) are saved using Pickle.

🛠️ Tech Stack
Language: Python

Web Framework: Streamlit

Machine Learning: Scikit-learn

Data Processing: Pandas, Numpy

Model Interpretation: LIME

Notebook Environment: Jupyter

📁 Project Structure
.
├── data/
│   ├── True.csv
│   └── Fake.csv
├── saved_model/
│   ├── model.pkl
│   └── vectorizer.pkl
├── Model_Training.ipynb
├── app.py
├── requirements.txt
└── README.md
data/: Contains the original dataset used for training.

saved_model/: Stores the trained model and vectorizer objects.

Model_Training.ipynb: The Jupyter Notebook for data preprocessing, model training, and evaluation.

app.py: The main script containing the Streamlit application code.

requirements.txt: A file listing the necessary Python packages.

README.md: The file you are currently reading.

🚀 Setup and Usage
1. Prerequisites
Python 3.8+ and Git must be installed.

2. Clone the Repository
Open your terminal and clone this repository to your local machine:

Bash

git clone <your_repository_url>
cd <repository_name>
3. Create a Virtual Environment & Install Dependencies
Create a virtual environment to avoid package conflicts.

Bash

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install the required packages from the requirements.txt file:

Bash

pip install -r requirements.txt
(If you do not have a requirements.txt file, create one with the following content):

Plaintext

pandas
numpy
scikit-learn
streamlit
lime
4. Train the Model
Open and run all cells in the Model_Training.ipynb notebook using Jupyter Notebook or VS Code.

Bash

jupyter notebook Model_Training.ipynb
This process will generate the saved_model/ directory containing model.pkl and vectorizer.pkl.

5. Run the Web Application
Once the model has been trained and saved, launch the Streamlit application:

Bash

streamlit run app.py
Open your web browser and navigate to http://localhost:8501 to start using the app.