import random

class Portfolio(object):
    def __init__(self,cash=0):
        self.cash = cash
        self.stock = {}
        self.mutualfund = {}
        self.history = []
        
    def addCash(self,cash):
        self.cash = self.cash + cash 
        self.history.append(f"{cash} of cash added to the account.\ncurrent balance: {self.cash}")
    
    def withdrawCash(self,cash):
        if cash >= self.cash:
            raise ValueError (f"Could not withdraw {cash}, since it is bigger than your current amount, {self.cash}. You have insufficient funds.")
        self.cash = self.cash - cash
        self.history.append(f"{cash} of cash withdrawn from the account.\ncurrent cash balance: {self.cash}")
        
    
    def buyStock(self, share, stocktype):
        
        if isinstance(share, int) = False :
                raise TypeError ("Can't do the operation. Please insert a whole amount.")     
        if share*stocktype.price > self.cash:
            raise ValueError (f"Could not complete the operation, since the {share*stocktype.price} is bigger than your current amount, {self.cash}. You have insufficient funds.")
        
            if stocktype.symbol in self.stock: 
                self.stock[stocktype.symbol] + share  
                #stocktype.symbol gives "number of shares" of that stocktype
            else: self.stock[stocktype.symbol] = {stocktype.symbol}
                self.stock[stocktype.symbol] = share
                
            self.cash = self.cash - share*stocktype.price
            self.history.append(f"{share*stocktype.price} of cash was used to buy stocks\ncurrent cash balance: {self.cash}")
            self.history.append(f"your current stock of type {stocktype}:", self.stock[stocktype.symbol], stocktype.symbol)
     

    def sellStock(self, share, stocktype):
        if isinstance(share, int) = False :
                raise TypeError ("Can't do the operation. Please insert a whole amount.")     
        if share > self.stock[stocktype.symbol]:
            raise ValueError("not enough stocks to sell")
        
        if stocktype.symbol in self.stock and share < self.stock[stocktype.symbol]:
            pass
        elif stocktype.symbol in self.stock and share == self.stock[stocktype.symbol]:
            del self.stock[stocktype.symbol]
            self.stock[stocktype.symbol] - share
        
        randomprice = share*stocktype.price*uniform(0.5,1.5)
        self.cash = self.cash + randomprice
        self.history.append(f"{randomprice} of cash was added to your account\ncurrent cash balance: {self.cash}")
        self.history.append(f"your current stock of type {stocktype}:", self.stock[stocktype.symbol], stocktype.symbol)
    
    
    def buyMutualFund(self, share, mftype): 
        
        if isinstance(share, int) = True :
                raise TypeError ("Can't do the operation. Please insert a fractional amount.")     
        if share*mftype.price > self.cash:
            raise ValueError (f"Could not complete the operation, since the {share*mftype.price} is larger than your current amount, {self.cash}. You have insufficient funds.")
        
        if mftype.symbol in self.mutualfund: 
            self.mutualfund[mftype.symbol] + share  
        else: self.mutualfund[mftype.symbol] = {mftype}
            self.mutualfund[mftype.symbol] = share
        
        self.cash = self.cash - share*mftype.price
        self.history.append(f"{share*mftype.price} of cash was used to buy mutualfunds\ncurrent cash balance: {self.cash}")
        self.history.append(f"your current mutual fund of type {mftype}:", self.mutualfund[mftype.symbol], mftype.symbol) 
        
    
    def sellMutualFund(self, share, mftype):
        
        if isinstance(share, int) = True :
                raise TypeError ("Can't do the operation. Please insert a fractional amount.")
        if share > self.mutualfund[mftype.symbol]:
            raise ValueError("not enough mutual funds to sell")
        if mftype.symbol in self.mutualfund and share < self.mutualfund[mftype.symbol]:
            pass
        elif stocktype.symbol in self.mutualfund and share == self.mutualfund[mftype.symbol]:
            del self.mutualfund[mftype.symbol]
            self.mutualfund[mftype.symbol] - share
            randomprice= share*uniform(0.8,1.2)
            self.cash = self.cash + share*randomprice
            self.history.append(f"{randomprice} of cash was added to your account\ncurrent cash balance: {self.cash}")
            self.history.append(f"your current mutualfund of type {mftype}:", self.mutualfund[mftype.symbol], mftype.symbol)

    def printport(self):
        print("--------------------------")
        print("Your current portfolio:")
        print("-cash:", self.cash)
        print("-stock:", self.stock)
        print("-mutual fund: \n", self.mutualfund )
        print("--------------------------")
        
    def history(self):
        text = '\n'.join(self.history)
        print("----------")
        print("Your transactions history:")
        print(text)
        print("----------")

 #shows inheritance and bond addition       
class Asset(object):
    def __init__(self, price, symbol):
        self.assettype = ""
        self.price = price
        self.symbol = symbol
        self.fractional = False
        
class Stock(Asset): 
    def __init__(self, price, symbol):
        Asset.__init__(self, price, symbol)
        self.symbol = symbol
        self.assettype = "Stock"
    
class MutualFund(Asset):
    def __init__(self, symbol):
        Asset.__init__(self, 1, symbol)
        self.assettype = "MutualFund"
        self.fractional = True
        
class Bond(Asset):
    def __init__(self, price):
        Asset.__init__(self, price, interest, symbol)
        self.assettype = "Bond"
        self.interest = interest
   
def trial():
    portfolio=Portfolio()
    portfolio.addCash(1000)
    portfolio.withdrawCash(50)
    s = Stock(20,"HFH")
    portfolio.buyStock(5, s)
    portfolio.sellStock(2, s)
    mf1 = MutualFund("BRT")
    mf2 = MutualFund("GHT")
    portfolio.buyMutualFund(10.3, mf1)
    portfolio.buyMutualFund(2, mf2)
    portfolio.sellMutualFund("BRT", 4)
    portfolio.sellMutualFund("GHT", 2)
    portfolio.printport()
    portfolio.history()
