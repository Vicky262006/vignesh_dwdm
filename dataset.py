import pandas as pd
from pymongo import MongoClient

# Load CSV
df = pd.read_csv("final.csv")

# Replace "NULL" strings with actual None
df = df.replace("NULL", None)

# Insert into MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["grocerydb"]
collection = db["orders"]

# Clear existing data (optional)
collection.delete_many({})
collection.insert_many(df.to_dict("records"))

print("Dataset loaded into MongoDB")
