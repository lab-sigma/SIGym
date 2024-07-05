from contract import Contract 
from bot import MMBot, TakerBot 
import math

# MM bots

class IMCMMBot(MMBot):
    def __init__(self, name, chips=0, maxspread=100000):
        # Call the constructor of the superclass MMBot
        super().__init__(name, chips, maxspread)

    def algorithm(self):
        # Override the implementation of common_function
        if self.info == 0:
            theo = 35
        else:
            allCardSum = 364 
            allCardNum = 52
            curCardSum = sum(self.info['Cards'])
            curCardNum = len(self.info['Cards'])
            theo = (allCardSum - curCardSum)/(allCardNum - curCardNum) * (5 - curCardNum) + curCardSum
            print("cardSum:{}, cardNum:{}, theo:{}".format(curCardSum, curCardNum, theo))
        print("theo:{}".format(theo))

        bid, ask, contracts = round(theo-2), round(theo+2), 5
        return [bid, ask, contracts]

class HRTMMBot(MMBot):
    def __init__(self, name, chips=0, maxspread=100000):
        # Call the constructor of the superclass MMBot
        super().__init__(name, chips, maxspread)

    def algorithm(self):
        # Override the implementation of common_function
        avg = 35
        if self.info == 0:
            theo = 35
            contractSize = 1 
        else:
            allCardSum = 364 
            allCardNum = 52
            curCardSum = sum(self.info['Cards'])
            curCardNum = len(self.info['Cards'])
            theo = (allCardSum - curCardSum)/(allCardNum - curCardNum) * (5 - curCardNum) + curCardSum
            contractSize = 10 - max(round ((theo-avg)/math.sqrt(allCardNum*14)), 9)
            print(curCardSum, curCardNum, theo)
        print(theo)

        bid, ask, contracts = round(theo-2), round(theo+2), contractSize
        return [bid, ask, contracts]

# Taker bots

class GoldmanTakerBot(TakerBot):
    def __init__(self, name, chips=0):
        # Call the constructor of the superclass MMBot
        super().__init__(name, chips)

    def algorithm(self):
        # Override the implementation of common_function
        theo = 35
        action = "h"
        curMarket = self.info['Actions'][-1]
        if theo > curMarket[0]['Market'][1]:
            action = "b"
        elif theo < curMarket[0]['Market'][0]:
            action = "s"
        return action

class TwoSigmaTakerBot(TakerBot):
    def __init__(self, name, chips=0):
        # Call the constructor of the superclass MMBot
        super().__init__(name, chips)

    def algorithm(self):
        # Override the implementation of common_function
        cards = self.info['Cards']
        theo = sum(cards) + 7 * (5-len(cards))
        action = "h"
        curMarket = self.info['Actions'][-1]
        if theo > curMarket[0]['Market'][1]:
            action = "b"
        elif theo < curMarket[0]['Market'][0]:
            action = "s"
        return action