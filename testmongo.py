import pymongo

client = pymongo.MongoClient("0.0.0.0:27017")
db = client["database"]
images = db["images"]

for x in images.find():
    print(x)