class CookieMaker:
    def __init__(self):
        self.result = 0

    def add(self, num):
        self.result = self.result + num
        return self.result
    
    def reset(self):
        self.result=0
        return self.result

cookie=CookieMaker()

cookie.result
cookie.add(3)
cookie.result

cookie.reset()
cookie.result