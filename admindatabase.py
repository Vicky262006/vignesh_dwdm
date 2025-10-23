import bcrypt
from pymongo import MongoClient

client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["grocerydb"]
admin_collection = db["admins"]

password = bcrypt.hashpw("vikram123".encode("utf-8"), bcrypt.gensalt())
admin_collection.insert_one({
    "name": "vikram",
    "email": "vikram@example.com",
    "password": password
})
print("Default admin created!")
