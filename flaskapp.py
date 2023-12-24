from flask import Flask, render_template, request
import pandas as pd
import joblib
import ast
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords if not already present
nltk.download('stopwords')

app = Flask(__name__)

# Load the clustering model
model = joblib.load('clustering_model.pkl')
print('\n Model loaded \n')

# Load the dataset
df = pd.read_csv("cleaned.csv")  # Replace with the path to your cleaned dataset
print('\n Dataset read \n')

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Evaluating rows of columns to their types
df['Features'] = df['Features'].apply(ast.literal_eval)
df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
df['directions'] = df['directions'].apply(ast.literal_eval)
print('\n Literal eval done \n')

# # Concatenate all strings in the 'Features' column into a single list
all_features = [feature for features in df['Features'] for feature in features]

# Remove stopwords from all_ingredients
nltk_stopwords = set(stopwords.words('english'))
more_stops = ['hello', 'hi', 'hey', 'good', 'morning', 'evening']  # Add your desired stop words here
stopwords = nltk_stopwords.union(more_stops)

all_features = [feature for feature in all_features if feature.lower() not in stopwords]
print('\n stopwords removed from all_ingredients \n')

# Initialize an empty list to store the documents
documents = df['RecipeCategory'].tolist()

# Create the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)
print('\n tfidf-matrox created \n')


def recommend_recipes(user_input):
    # Perform clustering on user input
    user_tfidf = vectorizer.transform([user_input])
    label = model.predict(user_tfidf)

    # Filter recipes based on the predicted cluster label
    cluster_labels = model.predict(tfidf_matrix)
    cluster_recipes = df[cluster_labels == label[0]]

    # Check if there are any recipes in the cluster
    if cluster_recipes.empty:
        return None

    # Calculate confidence scores for each recipe based on keyword matching
    recipe_scores = {}
    user_keywords = set(user_input.lower().split(','))

    for index, recipe in cluster_recipes.iterrows():
        features = set(recipe['Features'])
        matching_keywords = user_keywords.intersection(features)
        confidence_score = len(matching_keywords) / len(user_keywords)
        recipe_scores[index] = confidence_score

    # Sort recipes based on confidence scores in descending order
    sorted_recipes = sorted(recipe_scores, key=recipe_scores.get, reverse=True)

    # Select top 5 recipes with the highest confidence scores
    recommended_recipes = cluster_recipes.loc[sorted_recipes[:5]]

    return recommended_recipes


chat_history = []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['user_input']

    # Get recipe recommendations based on user input
    recommendations = recommend_recipes(user_input)

    # Check if the user input contains multiple words without a comma
    if len(user_input.split(',')) > 1 and ',' not in user_input:
        return render_template('recommendations.html',
                               chat_history=[{'USER': user_input,
                                             'BOT': 'Please separate multiple words with a comma.'}],
                               recommendations=pd.DataFrame().head())

    # Filter rows where any of the strings in 'Features' column contain the user input
    matching_keywords = []
    for word in user_input.split(','):
        if any(word.strip().lower() in feature.lower() for feature in all_features):
            matching_keywords.append(word)

    if not matching_keywords or any(word.strip().lower() in stopwords for word in user_input.split(',')):
        return render_template('recommendations.html',
                               chat_history=[{'USER': user_input,
                                             'BOT': 'No recipe found. Try again.'}],
                               recommendations=pd.DataFrame().head())

    # Get the recipe names for chat history display
    recipe_names = recommendations['title'].tolist()

    # Add the user input and recipe names to the chat history
    conversation = {'USER': user_input, 'BOT': dict(enumerate(recipe_names, 1))}

    chat_history.append(conversation)  # Add recommended recipes to chat history

    # Store the chat history in cookies (limited to the last 5 items)
    response = app.make_response(
        render_template('recommendations.html', recommendations=recommendations, chat_history=chat_history[-5:]))

    return response


if __name__ == '__main__':
    app.run()
