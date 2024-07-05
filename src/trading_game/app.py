from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route('/trading-game')
def trading_game():
    blah="hellooooooooo"
    return render_template('trading-game.html', cardsRevealed=[8, 1], cardsHidden=[5, 2, 3, 4])

@app.route('/')
def flask_page():
    blah="hellooooooooo"
    return render_template('trading-game.html', test=blah, cards=["test"])

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', person=name)