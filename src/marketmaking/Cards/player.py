class MM:

    def __init__(self, name, chips = 0, pos = 0):
        self.name = name
        self.chips = chips
        self.pos = pos
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
        if (bid < 0 or ask < 0 or bid >= ask or contracts <= 0):
            raise AssertionError("Invalid market")
        return [bid, ask, contracts]

    def liquidate(self, value):
        self.chips += self.pos * value
        self.pos = 0
        self.info = 0

    def __str__(self) -> str:
        return "Remaining chips: " + str(self.chips) + '\n' + "Current position: " + str(self.pos)

class Taker:

    def __init__(self, name, chips = 0, pos = 0):
        self.name = name
        self.chips = chips
        self.pos = pos
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
        self.chips += self.pos * value
        self.pos = 0
        self.info = 0

    def __str__(self) -> str:
        return ""