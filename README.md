# Name standardisation

### Setup
Run in jupyter environment and enable items from `requirements.txt`.

For use in docker environment, see https://github.com/viktorybloom/jupyter-stack and clone repos into `Jupyter Stack`.

### Overview

This name standardisation process utilises FuzzyWuzzy, and Affinity Propagation to cluster manually input records. 

Use cases include manually entered records across entites - businesses, items, names etc.

Large lists may encounter runtime slowdowns due to big o complexity - note I ran this over 100k+ records in approx 30min. 
