# -*- coding: utf-8 -*-

# import modules
import os

# MongoDB dependencies
from pymongo import MongoClient

# import base class
from .database import Database

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

    def get_collection(self, db_name: str, collection_name: str) -> None:
        '''
        Retrieves a collection from the MongoDB database.

        :param db_name: The name of the MongoDB database.
        :param collection_name: The name of the MongoDB collection.
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
