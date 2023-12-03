#MyGPT_SQL

"""
Python provides several libraries for working with databases and executing SQL queries. One of the most commonly used libraries is sqlite3 for SQLite databases, and sqlalchemy for a more general-purpose SQL toolkit that supports various database engines. Here, I'll provide examples for both.

Using sqlite3 (SQLite Database):
python
Copy code
"""

import sqlite3

# Connect to a SQLite database (or create a new one)
conn = sqlite3.connect('example.db')

# Create a cursor object
cursor = conn.cursor()

# Create a table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER
    )
''')

# Insert data into the table
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('John Doe', 25))

# Commit the changes and close the connection
conn.commit()
conn.close()

# Reconnect and query data
conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Execute a SELECT query
cursor.execute("SELECT * FROM users")

# Fetch the results
results = cursor.fetchall()

# Print the results
for row in results:
    print(row)

# Close the connection
conn.close()

"""
Using sqlalchemy (General SQL Toolkit):
python
Copy code
"""

from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table

# Create a SQLite database engine (or use a different database URL for other engines)
engine = create_engine('sqlite:///example.db')

# Create a table using sqlalchemy's MetaData and Table
metadata = MetaData()
users_table = Table(
    'users',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String),
    Column('age', Integer)
)

# Create the table in the database
metadata.create_all(engine)

# Insert data into the table
with engine.connect() as conn:
    conn.execute(users_table.insert().values(name='John Doe', age=25))

# Query data from the table
with engine.connect() as conn:
    result = conn.execute(users_table.select())
    results = result.fetchall()

# Print the results
for row in results:
    print(row)
#These examples cover basic database operations such as connecting to a database, creating tables, inserting data, and querying data. Modify the code according to your specific use case and database engine. If you're working with a different database (e.g., MySQL, PostgreSQL), you can adjust the connection URL and syntax accordingly.




