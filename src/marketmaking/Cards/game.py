import random
MAX = 10000

def GenerateCards(numCards):
    deck = [i for i in range(1,14) for _ in range(4)]
    selected_indices = random.sample(range(len(deck)), numCards)
    selected_cards = [deck[i] for i in selected_indices]
    return selected_cards

class Game:

    def __init__(self, mm, taker, numCards = 5):
        self.numCards = numCards
        self.cards = GenerateCards(numCards)
        self.mm = mm
        self.taker = taker
        self.bestBid = -1
        self.bestAsk = MAX 
        self.contracts = 0
        self.curRound = 0
        self.info = {"Cards": [], "Actions": [0]}

    def draw(self):
        self.info["Cards"].append(self.cards[self.curRound])
        self.info["Actions"].append([])
        self.curRound += 1
        print("ROUND", self.curRound)
        print("CARD DRAWN", self.cards[self.curRound-1])
    
    def send_info(self):
        self.mm.receive_info(self.info)
        self.taker.receive_info(self.info)

    def req_market(self):
        receive_object = self.mm.send_market()
        if type(receive_object) != list or len(receive_object) != 3:
            raise AssertionError ("Invalid Market")
        bid, ask, contracts = receive_object
        print("BID", bid)
        print("ASK", ask)
        print("CONTRACTS", contracts)
        if bid >= self.bestAsk or ask <= self.bestBid:
            raise AssertionError ("Crossed Market")

        self.bestBid = max(self.bestBid, bid)
        self.bestAsk = min(self.bestAsk, ask)
        self.contracts = contracts
        market_object = {"Market": [bid, ask]}
        self.info["Actions"][self.curRound].append(market_object)
        self.send_info()

    def req_taker(self):
        action = self.taker.send_action()
        print("action", action)
        if action == "BUY":
            self.mm.pos -= self.contracts
            self.taker.pos += self.contracts 
            self.mm.chips += self.bestAsk * self.contracts 
            self.taker.chips -= self.bestAsk * self.contracts
        elif action == "SELL":
            self.mm.pos += self.contracts
            self.taker.pos -= self.contracts
            self.mm.chips -= self.bestBid * self.contracts
            self.taker.chips += self.bestBid * self.contracts
        elif action != "HOLD":
            raise AssertionError ("invalid action")

        action_object = {"Taker": action}
        self.info["Actions"][self.curRound].append(action_object)
        self.send_info()

    def clear_book(self):
        self.bestBid = -1 
        self.bestAsk = MAX
        self.contracts = 0

    def end_game(self):
        self.mm.liquidate(sum(self.cards))
        self.taker.liquidate(sum(self.cards))