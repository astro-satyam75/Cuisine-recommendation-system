import time
import re
import pandas as pd
from nltk.tokenize import sent_tokenize

# starting time
start = time.time()

# Importing the dataset
df = pd.read_csv("D://SATYAM//new_bot//recipes.csv")  # Replace with the file path of your dataset
print('The shape of the dataset:', df.shape)
print('Columns of the dataset:', df.columns)
print('The first five elements of the dataset:', df.head())

# Cleaning and remodeling the dataset
df.drop_duplicates(inplace=True)
df.drop(columns=['RecipeYield', 'AggregatedRating', 'ReviewCount', 'RecipeServings'], inplace=True)
df.drop(columns=['AuthorId', 'AuthorName', 'DatePublished', 'PrepTime', "TotalTime", 'Images', 'CookTime', 'Calories',
                 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent',
                 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'],
        inplace=True)
df.dropna(inplace=True)

# Reforming the elements of the following columns to a readable format
df['Keywords'] = df["Keywords"].str.replace(r'c\(|\"|\"|\)|FreeOf...|<,', '', regex=True)
df['RecipeCategory'] = df["RecipeCategory"].str.replace(r'c\(|\"|\"|\)|FreeOf...|<,', '', regex=True)
df['Keywords'] = df['Keywords'].str.split(',')
df['RecipeIngredientQuantities'] = df["RecipeIngredientQuantities"].str.replace(r'c\(|\"|\"|\)', '', regex=True)
df['RecipeIngredientParts'] = df["RecipeIngredientParts"].str.replace(r'c\(|\"|\"|\)', '', regex=True)
df['RecipeInstructions'] = df["RecipeInstructions"].str.replace(r'c\(|\"|\"|\)|\n|,', '', regex=True)
df['RecipeIngredientQuantities'] = df['RecipeIngredientQuantities'].str.split(',')
df['RecipeIngredientParts'] = df['RecipeIngredientParts'].str.split(',')
df['RecipeInstructions'] = df['RecipeInstructions'].apply(lambda x: (sent_tokenize(x)))


# Function to convert a list into a dictionary
def list_to_dict(lst):
    return {i + 1: val for i, val in enumerate(lst)}


df['RecipeInstructions'] = df['RecipeInstructions'].apply(list_to_dict)

# Creating a new column Ingredients by combining RecipeIngredientParts and RecipeIngredientQuantities as a dictionary
df['Ingredients'] = df.apply(lambda row: dict(zip(row['RecipeIngredientParts'], row['RecipeIngredientQuantities'])),
                             axis=1)


# Function to remove leading and trailing whitespace from dictionary values
def remove_whitespace(d):
    return {k: v.strip() if isinstance(v, str) else v for k, v in d.items()}


df['Ingredients'] = df['Ingredients'].apply(remove_whitespace)

df['Keywords'] = df['Keywords'].apply(lambda x: [word.lstrip() for word in x])  # keywords was converted to list
# to remove whitespaces

df.drop(columns=['RecipeIngredientQuantities'], inplace=True)

df = df.loc[:, ['Name', 'RecipeCategory',
                'Keywords', 'RecipeIngredientParts',
                'Ingredients', 'RecipeInstructions']]

df.rename(columns={'Name': 'title', 'RecipeIngredientParts': 'ingredients_names', 'Ingredients': 'ingredients',
                   'RecipeInstructions': 'directions'}, inplace=True)

# Drop rows that don't begin with an alphabet
df = df[df['title'].str.match(r'^[a-zA-Z]')]
df = df[df['RecipeCategory'].str.match(r'^[a-zA-Z]')]

# Check if any rows have an empty ingredients list
no_ingredients = df['ingredients'] == {}

# Delete all rows where the ingredients list is empty
df = df.loc[~no_ingredients]

df['Features'] = df['Keywords'].apply(lambda x: [word for phrase in x for word in phrase.split(' ')]) \
                 + df['ingredients_names'].apply(lambda x: [word for phrase in x for word in phrase.split(' ')])

print('\n Features column created \n')

df['ingredients_names'] = df['ingredients_names'].apply(lambda x: [word for phrase in x for word in phrase.split(' ')])


def remove_non_alpha(df, column):
    # Create a regular expression to match non-alphabetical characters
    non_alpha_pattern = re.compile(r'[^a-zA-Z]')

    # Apply string cleaning operations to each element in the 'Features' column
    df[column] = df[column].apply(lambda x: [re.sub(non_alpha_pattern, '', item) for item in x])
    df[column] = df[column].apply(lambda x: [re.sub(r'http\S+', '', item) for item in x])

    return df


df = remove_non_alpha(df, 'Features')
df = remove_non_alpha(df, 'ingredients_names')

df.drop(columns=['Keywords'], inplace=True)

# Remove empty strings from each list in the column
df['ingredients_names'] = df['ingredients_names'].apply(lambda x: [item for item in x if item != ''])

df.to_csv('cleaned.csv')

end = time.time()
execution_time = end - start
print('Total runtime:', execution_time)