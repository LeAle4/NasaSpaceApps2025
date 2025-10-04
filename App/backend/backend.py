import pandas as pd
import parameters as p

class PredictionBatch():
    def __init__(self):
        self.BatchDataFrame = None
        
    
    def readCsvData(self, path: str):
        try:
            datafile = pd.read_csv(path, usecols=p.DATA_HEADERS)
        except Exception as e:
            print(e)
        
        self.batchDataFrame = datafile
        
        
        
class Exoplanet():
    def __init__(self):
        pass
    
if __name__ == '__main__':
    prediction_batch = PredictionBatch()
    prediction_batch.readCsvData("C:\\Users\\ADMIN\\Desktop\\Vicente\\Sistemas coheteria\\NasaSpaceApps2025\\Modelo\\koi_exoplanets.csv")
    
    #hola
    
