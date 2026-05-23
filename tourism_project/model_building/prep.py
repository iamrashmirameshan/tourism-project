
# for data manipulation
import pandas as pd

# for creating folders
import os

# for train-test split
from sklearn.model_selection import train_test_split

# for Hugging Face upload
from huggingface_hub import HfApi

# Initialize Hugging Face API
api = HfApi(token=os.getenv("MY_MLOps_Token"))

# Load dataset directly from Hugging Face
DATASET_PATH = "hf://datasets/rashmipr/tourism-project/tourism.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")

# Drop unnecessary columns
df.drop(columns=['CustomerID', 'Unnamed: 0'], inplace=True)

# Fix inconsistent category names

# Gender cleanup
df['Gender'] = df['Gender'].replace(
    'Fe Male',
    'Female'
)

# Marital Status cleanup
df['MaritalStatus'] = df['MaritalStatus'].replace(
    'Unmarried',
    'Single'
)

print("\nCleaned Gender Categories:")
print(df['Gender'].value_counts())

print("\nCleaned Marital Status Categories:")
print(df['MaritalStatus'].value_counts())

# Define target variable
target_col = 'ProdTaken'

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save processed datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("\nTrain-test split completed and files saved.")

# Upload processed files to Hugging Face
files = [
    "Xtrain.csv",
    "Xtest.csv",
    "ytrain.csv",
    "ytest.csv"
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="rashmipr/tourism-project",
        repo_type="dataset",
    )

print("\nProcessed datasets uploaded successfully.")
