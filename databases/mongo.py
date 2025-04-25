# -*- coding: utf-8 -*-

# import modules
import os

# import base class
from .database import Database

# MongoDB dependencies
from pymongo import MongoClient
from pymongo.collection import Collection

# MongoDBManager class
class MongoDBManager(Database):
    '''
    MongoDBManager class to manage MongoDB connections and operations.
    '''
    def __init__(self):
        '''
        Initializes the MongoDBManager instance.
        '''
        connection_string = 'mongodb://localhost:27017/'
        self.client = MongoClient(connection_string)

    def test_access_to_db_and_collection(self, db_name: str, collection_name: str) -> bool:
        '''
        Tests access to the specified database and collection.

        :param db_name: The name of the MongoDB database.
        :type db_name: str

        :param collection_name: The name of the MongoDB collection.
        :type collection_name: str

        :return: True if access is successful, False otherwise
        :rtype: bool
        '''
        db = self.client[db_name]

        # list all collections
        collections = db.list_collection_names()
        if collection_name in collections:
            return True
        else:
            return False

    def get_collection(self, db_name: str, collection_name: str) -> Collection:
        '''
        Retrieves a collection from the MongoDB database.

        :param db_name: The name of the MongoDB database.
        :type db_name: str

        :param collection_name: The name of the MongoDB collection.
        :type collection_name: str

        :return: The collection object.
        :rtype: pymongo.collection.Collection
        '''
        db = self.client[db_name]

        # access to db collection
        collection = db[collection_name]
        return collection
    
    def get_collected_uuids(self, db_name: str, collection_name: str) -> list:
        '''
        Retrieves all UUIDs from the MongoDB collection.

        :param db_name: The name of the MongoDB database.
        :param collection_name: The name of the MongoDB collection.

        :return: A list of UUIDs.
        '''
        db = self.client[db_name]
        collection = db[collection_name]
        uuids = collection.distinct('uuid')
        return uuids
