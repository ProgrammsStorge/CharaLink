import json
import random
import string

import flask
from flask import Flask, make_response
import sqlite3

app = Flask(__name__)

connection = sqlite3.connect('database.db')
cursor = connection.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Users (
    id INTEGER PRIMARY KEY,
    avatar TEXT,
    username TEXT NOT NULL,
    email TEXT NOT NULL,
    password TEXT NOT NULL,
    session_secret TEXT NOT NULL
    )
    ''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Characters (
    id INTEGER PRIMARY KEY,
    avatar TEXT,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    author_id INTEGER NOT NULL,
    tags TEXT NOT NULL,
    url TEXT NOT NULL
    )
    ''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Comments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    text TEXT NOT NULL,
    author_id INTEGER NOT NULL,
    to_id INTEGER NOT NULL,
    likes INTEGER
    )
    ''')

cursor.execute('SELECT * FROM Characters')
users = cursor.fetchall()




# Выводим результаты
for user in users:
  print(user)
connection.close()
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/register_page',methods=["GET"])
def register_page():
    return flask.render_template("register.html")

@app.route('/register',methods=["POST"])
def register():

    connection=get_db_connection()
    cursor=connection.cursor()

    cursor.execute('SELECT * FROM Users WHERE email == ?', (flask.request.form.get('email'),))
    results = cursor.fetchall()
    if list(results)==[]:
        session_secret = "".join(f"{random.choice(string.ascii_letters)}" for _ in range(50))
        cursor.execute('INSERT INTO Users (username, email, password,session_secret) VALUES (?, ?, ?, ?)', (
            flask.request.form.get('name'), flask.request.form.get('email'), flask.request.form.get('password'),session_secret))
        connection.commit()
        connection.close()
        resp = make_response(flask.redirect("/home_page", code=302))
        resp.set_cookie('Session', session_secret)
        return resp
    else:
        return flask.redirect("/register_page", code=302)

@app.route('/login_page',methods=["GET"])
def login_page():
    return flask.render_template("login.html")
@app.route('/login',methods=["POST"])
def login():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM Users WHERE email == ? AND password == ?',(flask.request.form.get('email'),flask.request.form.get('password')))
    users = cursor.fetchall()
    for user in users:
        print(dict(user))
    if list(users) != []:
        resp = make_response(flask.redirect("/home_page", code=302))
        resp.set_cookie('Session', dict(users[0])["session_secret"])
        return resp
    else:
        return flask.redirect("/login_page", code=302)

@app.route('/home_page',methods=["GET"])
def home_page():
    search = flask.request.args.get("search")
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute('SELECT * FROM Characters')
    results = cursor.fetchall()
    ai_profiles=list(results)
    connection.close()
    for i,v in enumerate(ai_profiles):
        ai_profiles[i] = dict(v)
    if search!="" and search!=None:
        ai_profiles_search=[]
        for i in ai_profiles:
            if search.lower() in i["name"].lower() or search.lower() in json.loads(i["tags"].lower()):
                ai_profiles_search.append(i)

        ai_profiles=ai_profiles_search.copy()
    for i,v in enumerate(ai_profiles):
        ai_profiles[i]["tags"] = ", ".join(json.loads(ai_profiles[i]["tags"]))
        ai_profiles[i]["description"] = ai_profiles[i]["description"][:28*2]
        ai_profiles[i]["name"] = ai_profiles[i]["name"][:25]
    return flask.render_template("home.html",profiles=ai_profiles)


@app.route('/clim/',methods=["GET"])
def install_manager():
    search = flask.request.args.get("search")
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute('SELECT * FROM Characters')
    results = cursor.fetchall()
    ai_profiles = list(results)
    for i,v in enumerate(ai_profiles):
        ai_profiles[i] = dict(v)
    if search!=None:
        for i in ai_profiles:
            if search.lower() in i["name"].lower() or search.lower() in str(i["id"]):
                return i["url"]
    return json.dumps( ai_profiles )

@app.route('/',methods=["GET"])
def redirect_main():
    return flask.redirect("/home_page", code=302)

@app.route('/upload_page',methods=["GET"])
def upload_page():
    return flask.render_template("upload.html")

@app.route('/profile_page/<id>',methods=["GET"])
def profile_page(id):
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute('SELECT * FROM Characters WHERE id == ?',(id,))
    results = cursor.fetchall()

    cursor.execute('SELECT * FROM Users WHERE id == ?', (list(results)[0]["author_id"],))
    results2 = cursor.fetchall()

    cursor.execute('SELECT * FROM Comments WHERE to_id == ?', (id,))
    results3 = cursor.fetchall()
    connection.close()

    return flask.render_template("profile.html",profile=list(results)[0],create_by=list(results2)[0],comments=list(results3))

@app.route('/comments',methods=["POST"])
def comments():
    connection = get_db_connection()
    cursor = connection.cursor()
    if flask.request.cookies.get('Session', '') != "":

        cursor.execute('SELECT * FROM Users WHERE session_secret == ?',
                       (flask.request.cookies.get('Session', ''),))
        users = cursor.fetchall()
        for user in users:
            print(dict(user))

        if list(users) != []:
            cursor.execute('SELECT * FROM Comments WHERE author_id == ? AND to_id == ?',
                           (dict(users[0])["id"], flask.request.form.get('to_id') ,))
            com_from = cursor.fetchall()
            if len(com_from)==0:
                cursor.execute('INSERT INTO Comments (name, text, author_id,to_id,likes) VALUES (?, ?, ?, ?, ?)', (
                    dict(users[0])["username"], flask.request.form.get('text'), dict(users[0])["id"],
                    flask.request.form.get('to_id'), 0))
                connection.commit()
                connection.close()
            else:
                return flask.redirect("/profile_page/"+flask.request.form.get('to_id'), code=302)
            return flask.redirect("/profile_page/"+flask.request.form.get('to_id'), code=302)
        else:
            return flask.redirect("/register_page", code=302)
    else:
        return flask.redirect("/login_page", code=302)

@app.route('/upload',methods=["POST"])
def upload():

    connection=get_db_connection()
    cursor=connection.cursor()
    if flask.request.cookies.get('Session', '')!="":

        cursor.execute('SELECT * FROM Users WHERE session_secret == ?',
                       (flask.request.cookies.get('Session', ''),))
        users = cursor.fetchall()
        for user in users:
            print(dict(user))
        if list(users) != []:
            cursor.execute('INSERT INTO Characters (name, description, author_id,tags,url) VALUES (?, ?, ?, ?, ?)', (
                flask.request.form.get('name'), flask.request.form.get('description'),dict(users[0])["id"],json.dumps(flask.request.form.get('tags').split()),flask.request.form.get('url')))
            connection.commit()
            connection.close()
            return flask.redirect("/home_page", code=302)
        else:
            return flask.redirect("/register_page", code=302)
    else:
        return flask.redirect("/login_page", code=302)

if __name__ == '__main__':
    app.run(port=8081,debug=True)

