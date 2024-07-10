from flask import Flask
from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from werkzeug.exceptions import abort

import sys
import os

# Add the 'src' directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Game imports
from trading_game.game import Game 
from trading_game.player import MM, Taker
from trading_game.bot import MMBot, TakerBot
from trading_game.example_bots import SimpleMMBot, SimpleTakerBot, IMCMMBot, HRTMMBot, GoldmanTakerBot, TwoSigmaTakerBot

app = Flask(__name__)

# Initialize default game variables
mmTotal = 0
takerTotal = 0
marketPrice = 0
numChips = 100

# Players and bots
mm = IMCMMBot("IMC Trading", chips = numChips, maxspread=5)
#mm = MM("player mm", chips=100, maxspread=5)
#mm = HRTMMBot("Hudson River Trading", chips=100, maxspread=5)
taker = Taker("player taker", chips = numChips)
#taker = GoldmanTakerBot("GoldmanTakerBot", chips = 100)

# Game settings
numRounds = 10
timer = 60
game = None
gameStatus = "inactive"
gameLog = ""

@app.route("/trading-game", methods=("GET", "POST"))
def trading_game():
    global mmTotal
    global takerTotal
    global marketPrice
    global mm
    global taker
    global numRounds
    global timer
    global game
    global gameStatus
    global gameLog

    cardsRevealed = cardsHidden = []
    bid = ask = contracts = 0
    
    if request.method == "POST":
        if request.form["submit_btn"] == "Start Game":
            """ Initialize game """

            # Set up players and bots
            mm = SimpleMMBot("Simple MM Bot", chips = numChips, maxspread=5)
            #mm = MM("player mm", chips=100, maxspread=5)
            #mm = HRTMMBot("Hudson River Trading", chips=100, maxspread=5)
            taker = Taker("player taker", chips = numChips)
            #taker = GoldmanTakerBot("GoldmanTakerBot", chips = 100)

            # Game settings
            numRounds = int(request.form["num_rounds"])
            game = Game(mm, taker, numRounds)

            # Start game
            game.start_round()
            marketPrice = game.price
            cardsRevealed = game.cards[:game.curRound]
            cardsHidden = game.cards[game.curRound:]
            bid, ask, contracts = game.req_market()
            
            gameStatus = "active"
        elif (request.form["submit_btn"] == "End Game"):
            gameStatus = "inactive"
        elif (request.form["submit_btn"] == "New Game"):
            gameStatus = "inactive"
        else:
            # Process player action
            action = request.form["submit_btn"].lower()
            amount = int(request.form[action + "_amt"])
            
            print("action:{}".format(action))
            print("amt:{}".format(amount))
            
            # Simulate this round
            if game.curRound != numRounds:
                
                game.req_taker(action, amount)

                game.clear_book()
                game.print_info()

                game.end_round()

                print(mm.name, "CHIPS:", mm.chips)
                print(taker.name, "CHIPS:", taker.chips)

                if game.curRound != numRounds:
                    game.start_round()
                    bid, ask, contracts = game.req_market()
                else:
                    # End game after last round
                    game.end_game()
                    gameStatus = "ended"
                    gameLog = game.info['Actions']

                    print(mm.name, "CHIPS:", mm.chips)
                    print(taker.name, "CHIPS:", taker.chips)
                    mmTotal += mm.chips 
                    takerTotal += taker.chips 

                    print("MARKET MAKER PROFIT", round(mmTotal - 100, 3))
                    print("TAKER PROFIT", round(takerTotal - 100, 3))

            # Update cards display
            cardsRevealed = game.cards[:game.curRound]
            cardsHidden = game.cards[game.curRound:]
    
    return render_template('trading-game.html', 
                           gameStatus=gameStatus,
                           cardsRevealed=cardsRevealed, 
                           cardsHidden=cardsHidden, 
                           bid=bid, 
                           ask=ask,
                           marketPrice=marketPrice,
                           startingBudget = numChips,
                           mmChips = mm.chips,
                           takerChips = taker.chips,
                           gameLog = gameLog)

@app.route('/')
def flask_page():
    return render_template('trading-game.html', cards=["test"])

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', person=name)