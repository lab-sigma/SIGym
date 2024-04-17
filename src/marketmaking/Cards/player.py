from contract import Contract

class MM:

    def __init__(self, name, chips = 0, maxspread = 100000):
        self.name = name
        self.chips = chips
        self.maxspread = maxspread
        self.contracts = []
        self.info = 0 ## null 

    def receive_info(self, info):
        self.info = info

    def send_market(self):
        # TODO
        bid = int(input("Enter the bid: "))
        ask = int(input("Enter the ask: "))
        contracts = int(input("Enter the num of contracts: "))

        if not all(isinstance(num, int) for num in [bid, ask, contracts]):
            raise AssertionError("All inputs must be integers")
        if (bid < 0 or ask < 0 or bid >= ask or (ask-bid > self.maxspread) or contracts <= 0):
            raise AssertionError("Invalid market")
        return [bid, ask, contracts]

    def liquidate(self, value):
        for c in self.contracts:
            self.chips += c.pnl(value)
        self.info = 0

    def print_contracts(self):
        for c in self.contracts:
            print(c)

    def __str__(self) -> str:
        return "Remaining chips: " + str(self.chips)

class Taker:

    def __init__(self, name, chips = 0):
        self.name = name
        self.chips = chips
        self.contracts = []
        self.info = 0 ## null

    def receive_info(self, info):
        self.info = info

    def send_action(self):
        # TODO
        action = ""
        while not action:
            action = input("Enter BUY(b), SELL(s), or HOLD(h): ")
            if action not in ["BUY", "SELL", "HOLD", "b", "s", "h"]:
                print("Invalid action")
                action = ""
        return action

    def liquidate(self, value):
        for c in self.contracts:
            self.chips += c.pnl(value)
        self.info = 0

    def print_contracts(self):
        for c in self.contracts:
            print(c)

    def __str__(self) -> str:
        return ""