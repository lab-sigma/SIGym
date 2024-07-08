class Contract:
    def __init__(self, contract_type, spot, pos):
        # 's' for sell and 'b' for buy
        self.contract_type = contract_type
        self.spot = spot
        self.pos = pos # number of stocks/contracts bought

    def pnl(self, fair_price):
        print(self)
        print("fair_price:{}".format(fair_price))
        if self.contract_type == 'b':
            pnl = (fair_price - self.spot) * self.pos
            print("spot:{}".format(self.spot))
            print("pos:{}".format(self.pos))
            print("profit:{}".format(pnl))
            return pnl
        else:
            pnl = (self.spot - fair_price) * self.pos
            print("spot:{}".format(self.spot))
            print("pos:{}".format(self.pos))
            print("equation:{}".format((self.spot - fair_price) * self.pos))
            print("profit:{}".format(pnl))
            return pnl
    
    def __str__(self) -> str:
        if self.contract_type == 'b':
            return "Bought " + str(self.pos) + " contract(s) at " + str(self.spot)
        else:
            return "Sold " + str(self.pos) + " contract(s) at " + str(self.spot)