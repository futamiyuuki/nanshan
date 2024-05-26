# temp

import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Example data
data = {
    'user_id': ['U1', 'U2', 'U3', 'U4'],
    'item_id': ['I1', 'I2', 'I3', 'I4'],
    'rating': [3, 4, 2, 5],
    'age_group': [1, 2, 1, 2],  # Just an example: 1 for 18-25, 2 for 26-35, etc.
    'income_level': [3, 2, 3, 1]  # Example: 1 for high, 2 for medium, 3 for low
}

df = pd.DataFrame(data)

# Hypothetical function to adjust ratings based on demographics
def adjust_ratings(row):
    if row['age_group'] == 2:
        row['rating'] *= 1.1  # Hypothetical adjustment
    if row['income_level'] == 1:
        row['rating'] *= 1.1  # Hypothetical adjustment
    return row

# Apply the adjustment function
df = df.apply(adjust_ratings, axis=1)

# Load the dataset into Surprise
reader = Reader(rating_scale=(1, 5.5))  # Adjusted max rating due to potential increases
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Split data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use the SVD algorithm
model = SVD()

# Train the model on the adjusted training set
model.fit(trainset)

# Test the model
predictions = model.test(testset)

# Calculate and print the RMSE
accuracy.rmse(predictions)

# Example: Predicting a rating for a specific user-item pair
user_id = 'U1'
item_id = 'I4'
predicted_rating = model.predict(user_id, item_id).est
print(f"Predicted rating for {user_id} on {item_id}: {predicted_rating}")
