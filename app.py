import pandas as pd
import joblib
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
import matplotlib.pyplot as plt

# RUN THE LINE BELOW WHEN RUNNING ON ANACONDA, JUPYTER OR COLLAB
# pip install --upgrade threadpoolctl
# pip install fsspec
# RESTART KERNEL AFTER INSTALLING

# Remove warnings from console
import warnings
warnings.filterwarnings("ignore")

# starting time
start = time.time()

# Load the dataset
df = pd.read_csv("cleaned.csv")  # Replace with the path to your cleaned dataset
print('\n Dataset read \n')


# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Initialize an empty list to store the documents
documents = df['RecipeCategory'].tolist()

# Create the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)
print('\n Vectorizer fitted \n')

# Train the model in batches
batch_size = 5000
num_batches = len(df) // batch_size

# Set the number of clusters
x = df['RecipeCategory'].unique()
num_clusters = len(x) # Number of unique RecipeCategory values

# Create a MiniBatchmodel model
model = MiniBatchKMeans(n_clusters=num_clusters, batch_size=100)
print('\n Model created \n Beginning model training....... \n')

for i in range(num_batches):
    start_index = i * batch_size
    end_index = start_index + batch_size

    # Extract the batch data
    batch_data = tfidf_matrix[start_index:end_index]

    # Fit the batch data to the model
    if batch_data.shape[0] > 0:  # Check if batch_data is not empty
        model.partial_fit(batch_data)

    # Print the current iteration
    print(f"Iteration: {i + 1}/{num_batches}")

print('\n Model trained \n')

# After training the model, save it to a file
joblib.dump(model, 'clustering_model.pkl')
print('\n Model saved \n')

# Evaluate the model's performance
print("Predicting labels...")
labels = []
for i, sample in enumerate(tfidf_matrix):
    print(f"Processing sample {i + 1}/{tfidf_matrix.shape[0]}")
    label = model.predict(sample)
    labels.append(label)
print("Label prediction completed.\n\n\n")

num_batches_eval = 30  # Number of batches to compute the evaluation metrics
batch_size_eval = len(df) // num_batches_eval

# Initialize lists to store the scores
davies_bouldin_scores = []
calinski_harabasz_scores = []
silhouette_scores = []

# Calculate the Davies-Bouldin Index
print("******** Calculating Davies-Bouldin Index ********")
davies_bouldin = 0.0

for i in range(num_batches_eval):
    start_index = i * batch_size_eval
    end_index = start_index + batch_size_eval

    # Extract the batch data
    batch_data = tfidf_matrix[start_index:end_index]
    batch_labels = labels[start_index:end_index]

    # Compute Davies-Bouldin Index for the batch
    davies_bouldin += davies_bouldin_score(batch_data.toarray().astype(np.float16), batch_labels)
    davies_bouldin_scores.append(davies_bouldin_score(batch_data.toarray().astype(np.float16), batch_labels))

    # Print the current iteration
    print(f"Davies-Bouldin Iteration: {i + 1}/{num_batches_eval}")

# Average Davies-Bouldin Index over all batches
davies_bouldin /= num_batches_eval
print(f"Davies-Bouldin Index: {davies_bouldin}")
print("Davies-Bouldin calculation completed.")

# Calculate the Calinski-Harabasz Index
print("\n\n******** Calculating Calinski-Harabasz Index *********")
calinski_harabasz = 0.0

for i in range(num_batches_eval):
    start_index = i * batch_size_eval
    end_index = start_index + batch_size_eval

    # Extract the batch data
    batch_data = tfidf_matrix[start_index:end_index]
    batch_labels = labels[start_index:end_index]

    # Compute Calinski-Harabasz Index for the batch
    calinski_harabasz += calinski_harabasz_score(batch_data.toarray().astype(np.float16), batch_labels)
    calinski_harabasz_scores.append(calinski_harabasz_score(batch_data.toarray().astype(np.float16), batch_labels))

    # Print the current iteration
    print(f"Calinski-Harabasz Iteration: {i + 1}/{num_batches_eval}")

# Average Calinski-Harabasz Index over all batches
calinski_harabasz /= num_batches_eval
print(f"Calinski-Harabasz Index: {calinski_harabasz}")
print("Calinski-Harabasz calculation completed.\n\n")


# Visualize the Davies-Bouldin scores
plt.figure(figsize=(12, 6))
sns.lineplot(x=range(num_batches_eval), y=davies_bouldin_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Score')
plt.title('Davies-Bouldin Index Over Iterations')

# Add annotations for each score
for i, score in enumerate(davies_bouldin_scores):
    plt.annotate(f'{score:.2f}', (i, score), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()

# Visualize the Calinski-Harabasz scores
plt.figure(figsize=(12, 6))
sns.lineplot(x=range(num_batches_eval), y=calinski_harabasz_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Score')
plt.title('Calinski-Harabasz Index Over Iterations')

# Add annotations for each score
for i, score in enumerate(calinski_harabasz_scores):
    plt.annotate(f'{score:.2f}', (i, score), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()



# Print the metrics
print(f"Number of clusters: {num_clusters}")
print(f"Davies-Bouldin Index: {davies_bouldin}")
print(f"Calinski-Harabasz Index: {calinski_harabasz}")

end = time.time()
execution_time = end - start
print('Total runtime:', execution_time)
