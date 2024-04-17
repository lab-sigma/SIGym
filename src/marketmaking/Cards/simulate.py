from game import Game 
from player import MM, Taker

if __name__ == "__main__":
    mm = MM("Jane Street", chips=100, maxspread=5)
    taker = Taker("Citadel") 
    numRounds = 5
    game = Game(mm,taker,numRounds)

    for _ in range(numRounds):
        game.start_round()
        game.req_market()
        game.req_taker()
        game.clear_book()
        #game.print_info()
        game.end_round()

    game.end_game()
    print(mm.name, "CHIPS:", mm.chips)
    #print(taker.name, "CHIPS:", taker.chips)