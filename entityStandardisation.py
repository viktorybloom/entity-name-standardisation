import numpy as np
import pandas as pd
import re
from collections import Counter
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('Suppliers_List.csv')
# Rename the field you are looking to clean {'Column to clean': 'Input'}
df = df.rename(columns={'Contact Name': 'Input'})

stopwords = {'inc', 'ltd', 'llc', 'trade', 'trading', 'warehouse', 'group', 'limited', 'com', 'co', 'au', 'uk', 'org'}


# +
def clean_special_characters(txt):
    # Define a list of special characters to replace with a separator
    seps = [" ", ":", ";", ".", ",", "*", "#", "\n", "@", "|", "/", "\\", "-", "_", "?", "%", "!", "^", "(", ")"]
    # Use the first separator as the default
    default_sep = seps[0]
    
    # Replace all other special characters with the default separator
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    
    # Replace any instances of '+' with a comma
    txt = re.sub('\+', ',', txt)
    
    # Split the text on the default separator, remove any empty strings, and join the remaining strings with the separator
    temp_list = [i.strip() for i in txt.split(default_sep)]
    temp_list = [i for i in temp_list if i]
    return " ".join(temp_list)


def clean_stopword(txt):
    # Split the text into a list of words
    temp_list = txt.split(" ")
    # Remove any words that are in the stopwords list
    temp_list = [i for i in temp_list if i not in stopwords]
    # Join the remaining words into a cleaned text string
    return " ".join(temp_list)


def data_cleaning(data, inputCol=['Input'], dropForeign=True):
    # Drop any rows with missing values in the specified column
    data.dropna(subset=inputCol, inplace=True)
    # Reset the DataFrame index
    data = data.rename_axis('ID').reset_index()
    # Count the number of non-ASCII characters in the specified column for each row
    data['nonAscii_count'] = data[inputCol].apply(lambda x: sum([not c.isascii() for c in x]))
    
    if dropForeign:
        # Drop any rows with non-ASCII characters in the specified column
        data = data[data.nonAscii_count == 0]
    else:
        pass
    
    # Drop the nonAscii_count column
    data.drop('nonAscii_count', axis=1, inplace=True)
    # Make a copy of the DataFrame
    data_clean = data.copy()
    # Convert all text in the specified column to lowercase
    data_clean['cleanInput'] = data_clean[inputCol].apply(lambda x: x.lower())
    # Apply the clean_special_characters function to the cleaned text data
    data_clean['cleanInput'] = data_clean['cleanInput'].apply(clean_special_characters)
    # Apply the clean_stopword function to the cleaned text data
    data_clean['cleanInput'] = data_clean['cleanInput'].apply(clean_stopword)
    return data_clean


# +
def similarity_matching(fuzzInput):
    """
    Calculates the similarity scores between a list of names.
    Returns a matrix where the value in each cell [i,j] is the similarity score between name i and name j.
    """
    # Create a square matrix to store the similarity scores between names
    # Initialize all values to 100 to indicate that they have not been compared yet
    similarity_array = np.ones((len(fuzzInput), len(fuzzInput))) * 100

    # Iterate over each pair of names and calculate their similarity scores
    for i in range(1, len(fuzzInput)):
        for j in range(i):
            # Calculate the token set ratio and partial ratio of the names
            # Add small values to prevent division by zero errors
            s1 = fuzz.token_set_ratio(fuzzInput[i], fuzzInput[j]) + 0.000000000001
            s2 = fuzz.partial_ratio(fuzzInput[i], fuzzInput[j]) + 0.00000000001
            # Calculate the similarity score between the two names and store it in the matrix
            similarity_array[i][j] = 2 * s1 * s2 / (s1 + s2)

    # Fill in the upper triangle of the matrix with the corresponding scores
    for i in range(len(fuzzInput)):
        for j in range(i + 1, len(fuzzInput)):
            similarity_array[i][j] = similarity_array[j][i]

    # Set the diagonal to 100 to indicate that a name is identical to itself
    np.fill_diagonal(similarity_array, 100)

    # Return the similarity matrix
    return similarity_array


def similarity_clusters(data, inputCol='Input', dropForeign=True):
    """
    Clusters a DataFrame of input data based on their names.
    Returns a DataFrame with the input IDs, their corresponding clusters, and other columns from the original data.
    """
    # Clean the data using the data_cleaning() function
    data_clean = data_cleaning(data, inputCol='Input', dropForeign=dropForeign)

    # Get a list of input names and IDs from the cleaned data
    clean_names = data_clean.cleanInput.tolist()
    clean_ids = data_clean.ID.tolist()

    # Compute a similarity array between all pairs of names
    similarity_array = similarity_matching(clean_names)

    # Cluster the input names using Affinity Propagation
    clusters = AffinityPropagation(affinity='precomputed').fit_predict(similarity_array)

    # Create a DataFrame with the input IDs and their corresponding clusters
    df_clusters = pd.DataFrame(list(zip(clean_ids, clusters)), columns=['ID', 'cluster'])

    # Merge the cluster assignments with the original cleaned data
    df_sim = df_clusters.merge(data_clean, on='ID', how='left')

    # Return the merged DataFrame
    return df_sim


# -

def standardised_names(df_sim):
    df_sim = similarity_clusters(df_sim)
    # Create a dictionary to store standard names for each cluster
    standard_names = {}
    
    # Loop through each unique cluster in the dataframe
    for cluster in df_sim['cluster'].unique():        
        # Get a list of all the names in the cluster
        names = df_sim[df_sim['cluster'] == cluster].cleanInput.to_list()
        common_substrings = []
        
        # If there is more than one name in the cluster
        # Find all common substrings between name pairs using SequenceMatcher
        if len(names) > 1:
            for i in range(0, len(names)):
                for j in range(i+1, len(names)):
                    seqMatch = SequenceMatcher(None, names[i], names[j])
                    match = seqMatch.find_longest_match(0, len(names[i]), 0, len(names[j]))
                    
                    if (match.size != 0):
                        common_substrings.append(names[i][match.a: match.a + match.size].strip())
            
            # If common substrings are found
            n = len(common_substrings)
            if n > 0:
                # Get the mode of the common substrings
                counts = Counter(common_substrings)
                get_mode = dict(counts)
                mode = [k for k, v in get_mode.items() if v == max(list(counts.values()))]
                # Join the mode substrings into a standard name and add to dictionary
                standard_names[cluster] = ";".join(mode)
            # If no common substrings are found, use the first name in the cluster as the standard name
            else:
                standard_names[cluster] = names[0]
        # If there is only one name in the cluster, use that as the standard name
        else:
            standard_names[cluster] = names[0]
    
    # Create a dataframe from the dictionary of standard names
    df_standard_names = pd.DataFrame(list(standard_names.items()), columns=['cluster', 'standard_name'])
    
    # Merge the standard names dataframe with the original dataframe
    df_sim = df_sim.merge(df_standard_names, on='cluster', how='left')
    
    # Calculate a fuzzy matching score between the standard name and each name in the cluster
    df_sim['similarity_score'] = df_sim.apply(lambda x: fuzz.token_set_ratio(x['standard_name'], x['cleanInput']), axis=1)
    
    # Remove spaces from standard names and check for duplicate names
    df_sim['standard_name_clean'] = df_sim['standard_name'].apply(lambda x: x.replace(" ", ""))
    for name in df_sim['standard_name_clean'].unique():
        if len(df_sim[df_sim['standard_name_clean'] == name]['cluster'].unique()) > 1:
            # If a duplicate name is found, use the name as the standard name for all clusters with that name
            df_sim.loc[df_sim['standard_name_clean'] == name, 'standard_name'] = name
    
    # Drop the temporary column used to check for duplicate names and return the final dataframe
    return df_sim.drop('standard_name_clean', axis=1)


# +
# Check if data_cleaning works
#cleaned_data = data_cleaning(df)

# Check if similarity_clusters works
#cleaned_data = similarity_clusters(df)

cleaned_data = standardised_names(df)
cleaned_data.to_csv('clean_names.csv', index=False)

