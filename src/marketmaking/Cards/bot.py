from contract import Contract

class MMBot:

    def __init__(self, name, chips = 0, maxspread = 100000):
        self.name = name
        self.chips = chips
        self.maxspread = maxspread
        self.contracts = []
        self.info = 0 ## null 

    def receive_info(self, info):
        self.info = info

    def algorithm(self):
        # TO OVERRIDE
        bid, ask, contracts = 33, 37, 5
        # TO OVERRIDE

        return [bid, ask, contracts]

    def send_market(self):
        bid, ask, contracts = self.algorithm()

        if not all(isinstance(num, int) for num in [bid, ask, contracts]):
            raise ValueError("All inputs must be integers")
        if bid < 0 or ask < 0 or bid >= ask or (ask - bid > self.maxspread) or contracts <= 0:
            raise ValueError("Invalid market")
            
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

class TakerBot:

    def __init__(self, name, chips = 0):
        self.name = name
        self.chips = chips
        self.contracts = []
        self.info = 0 ## null

    def receive_info(self, info):
        self.info = info

    def algorithm(self):
        # TODO 
        action = "b"
        # TODO

        return action 

    def send_action(self):
        action = ""
        while not action:
            # TODO
            action = self.algorithm()
            # TODO 

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