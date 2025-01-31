import pandas as pd
import random

# Read the CSV file
file_path = 'resnet18_places_imagenet_broden.csv'
df = pd.read_csv(file_path)

# Sort the rows by the similarity column in descending order
df_sorted = df.sort_values(by='similarity', ascending=False)

# Divide the data into sections
top_200 = df_sorted.head(200)
bottom_200 = df_sorted.tail(200)
middle_section = df_sorted.iloc[(len(df_sorted) // 2 - 100):(len(df_sorted) // 2 + 100)]

# Sample uniformly at random 3 elements from each section
top_samples = top_200.sample(n=3, random_state=42)
bottom_samples = bottom_200.sample(n=3, random_state=42)
middle_samples = middle_section.sample(n=3, random_state=42)

# Extract the descriptions and unit numbers into the required format
result_dict = {
    "descriptions": {
        "top": top_samples['description'].tolist(),
        "middle": middle_samples['description'].tolist(),
        "bottom": bottom_samples['description'].tolist(),
    },
    "units": {
        "top": top_samples['unit'].tolist(),
        "middle": middle_samples['unit'].tolist(),
        "bottom": bottom_samples['unit'].tolist(),
    }
}

# Print the results
print(result_dict)
