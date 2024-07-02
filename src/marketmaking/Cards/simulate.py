from game import Game 
from player import MM, Taker
from bot import MMBot, TakerBot
from example_bots import IMCMMBot, HRTMMBot, GoldmanTakerBot, TwoSigmaTakerBot

if __name__ == "__main__":
    numGames = 1
    mmTotal = 0
    takerTotal = 0
    for _ in range(numGames):

        mm = IMCMMBot("IMC Trading", chips=100, maxspread=5)
        #mm = MM("me", chips=100, maxspread=5)
        #mm = HRTMMBot("Hudson River Trading", chips=100, maxspread=5)
        #taker = GoldmanTakerBot("Goldman Sachs", chips=100) 
        taker = TwoSigmaTakerBot("Two Sigma", chips=100)
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
        print(mm.name, "CHIPS:", mm.chips)
        print(taker.name, "CHIPS:", taker.chips)
        mmTotal += mm.chips 
        takerTotal += taker.chips 
    
    print("MARKET MAKER AVG PROFIT", round(mmTotal/numGames - 100,3))
    print("TAKER AVG PROFIT", round(takerTotal/numGames - 100,3))