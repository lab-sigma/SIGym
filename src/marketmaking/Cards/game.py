import random
from contract import Contract

MAX = 10000

def GenerateCards(numCards):
    deck = [i for i in range(1,14) for _ in range(4)]
    selected_indices = random.sample(range(len(deck)), numCards)
    selected_cards = [deck[i] for i in selected_indices]
    return selected_cards, selected_indices

class Game:

    def __init__(self, mm, taker, numCards = 5):
        self.numCards = numCards
        self.cards, self.cInd = GenerateCards(numCards)
        self.mm = mm
        self.taker = taker
        self.bestBid = -1
        self.bestAsk = MAX 
        self.contracts_available = 0
        self.curRound = 0
        self.info = {"Cards": [], "Actions": []}
        self.symbols = {0:'♠',1:'♥',2:'♣',3:'♦'}
        self.values = {1:"A",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"10",11:"J",12:"Q",13:"K"}

    @property
    def price(self):
        return sum(self.cards)

    #this is just a lazy and horrible approx for now
    def get_fair_price_range(self):
        return self.numCards, self.numCards * 13

    def print_board(self):
        print("-"*12+"ROUND "+str(self.curRound+1)+"-"*12)
        board = ""
        for i in range(self.curRound):
            board += self.symbols[self.cInd[i] // 13] + self.values[self.cards[i]] + " "
        for _ in range(self.curRound+1, self.numCards+1):
            board += "? "
        print("CARD(s) DRAWN:", board)
    
    def start_round(self):
        self.info["Cards"].append(self.cards[self.curRound])
        self.info["Actions"].append([])
        self.print_board()

    def end_round(self):
        self.curRound += 1

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
        self.contracts_available = contracts
        market_object = {"Market": [bid, ask]}
        self.info["Actions"][self.curRound].append(market_object)
        self.send_info()

    def req_taker(self):
        action = self.taker.send_action()
        if action == "BUY" or action == 'b':
            self.mm.contracts.append(Contract('s', self.bestAsk, self.contracts_available))
            self.taker.contracts.append(Contract('b', self.bestAsk, self.contracts_available))
        elif action == "SELL" or action == 's':
            self.mm.contracts.append(Contract('b', self.bestBid, self.contracts_available))
            self.taker.contracts.append(Contract('s', self.bestBid, self.contracts_available))

        action_object = {"Taker": action}
        self.info["Actions"][self.curRound].append(action_object)
        self.send_info()

    def clear_book(self):
        self.bestBid = -1 
        self.bestAsk = MAX
        self.contracts_available = 0

    def print_info(self):
        print(self.mm)
        print(self.taker)

    def end_game(self):
        self.print_board()
        self.mm.print_contracts()
        self.mm.liquidate(self.price)
        self.taker.liquidate(self.price)