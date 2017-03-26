import json


class Config:
    def __init__(self,path):
        self.path = path
        self.config = self.__read__()
      
    
    def __read__(self):
        with open(self.path,'r') as f:
            js = json.loads(f.read())
        
        return js
    
    
    def get(self):
        
        return self.config
    