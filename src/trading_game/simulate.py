import sys
import os

# Add the 'src' directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Imports
from trading_game.game import Game 
from trading_game.player import MM, Taker
from trading_game.bot import MMBot, TakerBot
from trading_game.example_bots import SimpleMMBot, SimpleTakerBot, IMCMMBot, HRTMMBot, GoldmanTakerBot, TwoSigmaTakerBot

if __name__ == "__main__":
    numGames = 10
    mmTotal = 0
    takerTotal = 0

    for _ in range(numGames):
        mm = IMCMMBot("IMC Trading", chips=100, maxspread=5)
        mm = SimpleMMBot("Simple MM Bot", chips = 100, maxspread=5)
        #mm = MM("player mm", chips=100, maxspread=5)
        #mm = HRTMMBot("Hudson River Trading", chips=100, maxspread=5)
        #taker = GoldmanTakerBot("Goldman Sachs", chips=100) 
        #taker = TwoSigmaTakerBot("Two Sigma", chips=100)
        taker = Taker("player taker", chips=100)
        taker = SimpleTakerBot("Simple Taker Bot", chips = 100)
        numRounds = 10
        game = Game(mm,taker,numRounds)

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
        print("info actions:")
        print(game.info['Actions'])
        print(mm.name, "CHIPS:", mm.chips)
        print(taker.name, "CHIPS:", taker.chips)
        mmTotal += mm.chips 
        takerTotal += taker.chips 
    
    print("MARKET MAKER PROFIT", round(mmTotal/numGames - 100,3))
    print("TAKER PROFIT", round(takerTotal/numGames - 100,3))