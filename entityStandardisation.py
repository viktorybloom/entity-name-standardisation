import numpy as np
import pandas as pd
import re
from collections import Counter
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('List.csv')
# Rename the field you are looking to clean {'Column to clean': 'Input'}
df = df.rename(columns={'ColumnOfDataToClean': 'Input'})

# Example stopwords to remove
stopwords = {'inc', 'ltd', 'llc', 'trade', 'trading', 'warehouse', 'group', 'limited', 'com', 'co', 'au', 'uk', 'org'}


# Initial Cleanup of input column

def clean_special_characters(txt):
    seps = [" ", ":", ";", ".", ",", "*", "#", "\n", "@", "|", "/", "\\", "-", "_", "?", "%", "!", "^", "(", ")"]
    # Use the first separator as the default ' '
    default_sep = seps[0]
    
    # Replace all other special characters with the default separator
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    
    txt = re.sub('\+', ',', txt)
    
    temp_list = [i.strip() for i in txt.split(default_sep)]
    temp_list = [i for i in temp_list if i]
    return " ".join(temp_list)

def clean_stopword(txt):
    temp_list = txt.split(" ")
    temp_list = [i for i in temp_list if i not in stopwords]
    return " ".join(temp_list)

def data_cleaning(data, inputCol=['Input'], dropForeign=True):
    data.dropna(subset=inputCol, inplace=True)
    data = data.rename_axis('ID').reset_index()
    data['nonAscii_count'] = data[inputCol].apply(lambda x: sum([not c.isascii() for c in x]))
    
    if dropForeign:
        data = data[data.nonAscii_count == 0]
    else:
        pass
   # Final cleaning of initial input column 
    data.drop('nonAscii_count', axis=1, inplace=True)
    data_clean = data.copy()
    data_clean['cleanInput'] = data_clean[inputCol].apply(lambda x: x.lower())
    data_clean['cleanInput'] = data_clean['cleanInput'].apply(clean_special_characters)
    data_clean['cleanInput'] = data_clean['cleanInput'].apply(clean_stopword)
    return data_clean


# beware of Big O, will look for optimisations to this algorithm 
def similarity_matching(fuzzInput):
    """
    Calculates the similarity scores between a list of names.
    Returns a matrix where the value in each cell [i,j] is the similarity score between name i and name j.
    """
    similarity_array = np.ones((len(fuzzInput), len(fuzzInput))) * 100

    # Iterate over each pair of names and calculate their similarity scores
    for i in range(1, len(fuzzInput)):
        for j in range(i):
            s1 = fuzz.token_set_ratio(fuzzInput[i], fuzzInput[j]) + 0.000000000001
            s2 = fuzz.partial_ratio(fuzzInput[i], fuzzInput[j]) + 0.00000000001
            similarity_array[i][j] = 2 * s1 * s2 / (s1 + s2)

    # Fill in the upper triangle of the matrix with the corresponding scores
    for i in range(len(fuzzInput)):
        for j in range(i + 1, len(fuzzInput)):
            similarity_array[i][j] = similarity_array[j][i]

    np.fill_diagonal(similarity_array, 100)

    return similarity_array


def similarity_clusters(data, inputCol='Input', dropForeign=True):
    """
    Clusters a DataFrame of input data based on their names.
    Returns a DataFrame with the input IDs, their corresponding clusters, and other columns from the original data.
    """
    data_clean = data_cleaning(data, inputCol='Input', dropForeign=dropForeign)

    clean_names = data_clean.cleanInput.tolist()
    clean_ids = data_clean.ID.tolist()

    similarity_array = similarity_matching(clean_names)

    # Cluster the input names using Affinity Propagation
    clusters = AffinityPropagation(affinity='precomputed').fit_predict(similarity_array)

    df_clusters = pd.DataFrame(list(zip(clean_ids, clusters)), columns=['ID', 'cluster'])

    df_sim = df_clusters.merge(data_clean, on='ID', how='left')

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
                standard_names[cluster] = ";".join(mode)
            # If no common substrings are found, use the first name in the cluster as the standard name
            else:
                standard_names[cluster] = names[0]
        # If there is only one name in the cluster, use that as the standard name
        else:
            standard_names[cluster] = names[0]
    
    # Create a dataframe from the dictionary of standard names
    df_standard_names = pd.DataFrame(list(standard_names.items()), columns=['cluster', 'standard_name'])
    
    df_sim = df_sim.merge(df_standard_names, on='cluster', how='left')
    
    # Calculate a fuzzy matching score between the standard name and each name in the cluster
    df_sim['similarity_score'] = df_sim.apply(lambda x: fuzz.token_set_ratio(x['standard_name'], x['cleanInput']), axis=1)
    
    df_sim['standard_name_clean'] = df_sim['standard_name'].apply(lambda x: x.replace(" ", ""))
    for name in df_sim['standard_name_clean'].unique():
        if len(df_sim[df_sim['standard_name_clean'] == name]['cluster'].unique()) > 1:
            # If a duplicate name is found, use the name as the standard name for all clusters with that name
            df_sim.loc[df_sim['standard_name_clean'] == name, 'standard_name'] = name
    
    return df_sim.drop('standard_name_clean', axis=1)


# +
# Check if data_cleaning works
#cleaned_data = data_cleaning(df)

# Check if similarity_clusters works
#cleaned_data = similarity_clusters(df)

cleaned_data = standardised_names(df)
cleaned_data.to_csv('clean_names.csv', index=False)

