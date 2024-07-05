from flask import render_template
from trading_game import app

@app.route('/trading_game')
def flask_page():
    return render_template('trading_game.html')