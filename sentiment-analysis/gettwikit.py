import asyncio
from twikit import Client, TooManyRequests
import time
from datetime import datetime
import csv
from configparser import ConfigParser
from random import randint


#* login credentials
config = ConfigParser()
config.read('config.ini')
username = config['X']['username']
email = config['X']['email']
password = config['X']['password']


#* authenticate to X.com
#! 1) use the login credentials. 2) use cookies.
client = Client(language='en-US')
asyncio.run(client.login(auth_info_1=username, auth_info_2=email, password=password))
asyncio.run(client.save_cookies('cookies.json'))





