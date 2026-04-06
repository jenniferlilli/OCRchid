from dotenv import load_dotenv
import os

load_dotenv()

database_url = os.getenv("DATABASE_URL")
if database_url:
    print("DATABASE_URL found:", database_url)
else:
    print("DATABASE_URL NOT found.")
from dotenv import load_dotenv
import os

load_dotenv()

database_url = os.getenv("DATABASE_URL")
if database_url:
    print("DATABASE_URL found:", database_url)
else:
    print("DATABASE_URL NOT found.")

