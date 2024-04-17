class Contract:
    def __init__(self, contract_type, spot, pos):
        # 's' for sell and 'b' for buy
        self.contract_type = contract_type
        self.spot = spot
        self.pos = pos

    def pnl(self, fair_price):
        if self.contract_type == 'b':
            return (fair_price - self.spot) * self.pos
        else:
            return (self.spot - fair_price) * self.pos
    
    def __str__(self) -> str:
        if self.contract_type == 'b':
            return "Bought " + str(self.pos) + " contract(s) at " + str(self.spot)
        else:
            return "Sold " + str(self.pos) + " contract(s) at " + str(self.spot)