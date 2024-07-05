from flask import Flask
from flask import render_template

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
taker = Taker("player taker", chips=100)
numRounds = 6
game = Game(mm,taker,numRounds)
cards = game.cards

"""
for _ in range(numRounds):
    game.start_round()
    game.req_market()
    game.req_taker()
    game.clear_book()
    game.print_info()
    game.end_round()
    print(mm.name, "CHIPS:", mm.chips)
    print(taker.name, "CHIPS:", taker.chips)

game.end_game()
print(mm.name, "CHIPS:", mm.chips)
print(taker.name, "CHIPS:", taker.chips)
mmTotal += mm.chips 
takerTotal += taker.chips 

print("MARKET MAKER PROFIT", round(mmTotal - 100, 3))
print("TAKER PROFIT", round(takerTotal - 100, 3))

"""

@app.route('/trading-game')
def trading_game():
    return render_template('trading-game.html', cardsRevealed=cards, cardsHidden=[5, 2, 3, 4])

@app.route('/')
def flask_page():
    return render_template('trading-game.html', cards=["test"])

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', person=name)