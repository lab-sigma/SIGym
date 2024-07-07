from flask import Flask
from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from werkzeug.exceptions import abort

from .game import Game 
from .player import MM, Taker
from .bot import MMBot, TakerBot
from .example_bots import IMCMMBot, HRTMMBot, GoldmanTakerBot, TwoSigmaTakerBot

app = Flask(__name__)

# Initialize game
mmTotal = 0
takerTotal = 0

mm = IMCMMBot("IMC Trading", chips=100, maxspread=5)
#mm = MM("player mm", chips=100, maxspread=5)
#mm = HRTMMBot("Hudson River Trading", chips=100, maxspread=5)
taker = Taker("player taker", chips = 100)
taker = GoldmanTakerBot("GoldmanTakerBot", chips = 100)
numRounds = 6
game = Game(mm,taker,numRounds)
cardsRevealed = game.cards[:game.curRound]
cardsHidden = game.cards[game.curRound:]

@app.route("/trading-game", methods=("GET", "POST"))
def trading_game():
    global cardsRevealed
    global cardsHidden
    global mmTotal
    global takerTotal

    if request.method == "POST":
        # Process player action
        amount = 0
        if request.form["submit_btn"] == "Buy":
            amount = request.form["amt_bought"]
        elif request.form["submit_btn"] == "Sell":
            amount = request.form["amt_sold"]
        
        print("amt:{}".format(amount))
        
        # Simulate this round
        if game.curRound != numRounds:
            game.start_round()
            game.req_market()
            game.req_taker()
            game.clear_book()
            game.print_info()
            game.end_round()
            print(mm.name, "CHIPS:", mm.chips)
            print(taker.name, "CHIPS:", taker.chips)
        else:
            # End game after last round
            game.end_game()
            print(mm.name, "CHIPS:", mm.chips)
            print(taker.name, "CHIPS:", taker.chips)
            mmTotal += mm.chips 
            takerTotal += taker.chips 

            print("MARKET MAKER PROFIT", round(mmTotal - 100, 3))
            print("TAKER PROFIT", round(takerTotal - 100, 3))

        # Update cards display
        cardsRevealed = game.cards[:game.curRound]
        cardsHidden = game.cards[game.curRound:]
    return render_template('trading-game.html', cardsRevealed=cardsRevealed, cardsHidden=cardsHidden)

@app.route('/')
def flask_page():
    return render_template('trading-game.html', cards=["test"])

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', person=name)