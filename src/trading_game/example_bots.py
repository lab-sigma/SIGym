from .contract import Contract 
from .bot import MMBot, TakerBot 
import math

def CalculateTotalCardSum(numSuits, cardsPerSuit):
    suit = [i for i in range(1, cardsPerSuit + 1)]
    suitSum = sum(suit)
    return suitSum * numSuits

""" MM bots """

class SimpleMMBot(MMBot):
    def __init__(self, name, chips=0, maxspread=100000):
        # Call the constructor of the superclass MMBot
        super().__init__(name, chips, maxspread)

    def algorithm(self):
        # Override the implementation of common_function
        numSuits = self.info["NumSuits"]
        cardsPerSuit = self.info["CardsPerSuit"]
        allCardSum = CalculateTotalCardSum(numSuits, cardsPerSuit) 
        allCardNum = numSuits * cardsPerSuit # total deck size
        n = self.info["NumCards"] # number of rounds (total #cards drawn from deck)
        print("info cards:")
        print(self.info['Cards'])

        if self.info == 0:
            # Expected value for set of n unknown cards 
            # 42 for 6 cards
            theo = (allCardSum / allCardNum) * n
        else:
            curCardSum = sum(self.info['Cards'])
            curCardNum = len(self.info['Cards']) # number of cards revealed

            remainingSum = allCardSum - curCardSum
            remainingNum = allCardNum - curCardNum
            numHiddenCards = self.info["NumCards"] - curCardNum

            theo = (remainingSum / remainingNum) * numHiddenCards + curCardSum
            print("----mm---- \n cardSum:{}, cardNum:{}, theo:{}".format(curCardSum, curCardNum, theo))
        print("mm theo:{}".format(theo))

        bid, ask, contracts = round(theo-2), round(theo+2), 5

        # Set bid = ask = expected value
        #bid, ask, contracts = round(theo), round(theo), 5
        return [bid, ask, contracts]

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

""" Taker bots """

# disadvantaged for when mm has 1 extra card knowledge
class SimpleTakerBot(TakerBot):
    def __init__(self, name, chips=0):
        # Call the constructor of the superclass MMBot
        super().__init__(name, chips)

    def algorithm(self):
        # Override the implementation of common_function
        numSuits = self.info["NumSuits"]
        cardsPerSuit = self.info["CardsPerSuit"]
        allCardSum = CalculateTotalCardSum(numSuits, cardsPerSuit) 
        allCardNum = numSuits * cardsPerSuit # total deck size
        n = self.info["NumCards"] # number of rounds (total #cards drawn from deck)
        print("info cards:")
        print(self.info['Cards'])

        if self.info == 0 or len(self.info['Cards']) == 1:
            # Expected value for set of n unknown cards 
            # 42 for 6 cards
            theo = (allCardSum / allCardNum) * n
        else:
            curCardSum = sum(self.info['Cards']) - self.info['Cards'][len(self.info['Cards']) - 1]
            curCardNum = len(self.info['Cards']) - 1 # number of cards revealed

            remainingSum = allCardSum - curCardSum
            remainingNum = allCardNum - curCardNum
            numHiddenCards = self.info["NumCards"] - curCardNum

            theo = (remainingSum / remainingNum) * numHiddenCards + curCardSum
            print("----taker---- \n cardSum:{}, cardNum:{}, theo:{}".format(curCardSum, curCardNum, theo))
        print("taker theo:{}".format(theo))

        action = "h"
        curMarket = self.info['Actions'][-1]

        if theo > curMarket[0]['Market'][1]:
            # Buy if ask price is less than market price
            action = "b"
        elif theo < curMarket[0]['Market'][0]:
            # Sell if bid price is more than market price 
            action = "s"
        return action

# Calculate initial expected value of true price and continues
# playing based off this value every round
class SuperSimpleTakerBot(TakerBot):
    def __init__(self, name, chips=0):
        # Call the constructor of the superclass MMBot
        super().__init__(name, chips)

    def algorithm(self):
        # Override the implementation of common_function
        numSuits = self.info["NumSuits"]
        cardsPerSuit = self.info["CardsPerSuit"]
        allCardSum = CalculateTotalCardSum(numSuits, cardsPerSuit) 
        allCardNum = numSuits * cardsPerSuit # total deck size
        n = self.info["NumCards"] # total number of cards drawn from deck
        
        print("info cards:")
        print(self.info['Cards'])

        # Expected value for set of n unknown cards 
        # 42 for 6 cards
        theo = (allCardSum / allCardNum) * n
        print("taker theo:")
        print(theo)

        action = "h"
        curMarket = self.info['Actions'][-1]

        if theo > curMarket[0]['Market'][1]:
            # Buy if ask price is less than market price
            action = "b"
        elif theo < curMarket[0]['Market'][0]:
            # Sell if bid price is more than market price 
            action = "s"
        return action

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