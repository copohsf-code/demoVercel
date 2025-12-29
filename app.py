import os
import psycopg2
from flask import Flask, jsonify

app = Flask(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

@app.route("/users")
def get_users():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT id, email FROM users;")
    users = cur.fetchall()

    cur.close()
    conn.close()

    return jsonify(users)
