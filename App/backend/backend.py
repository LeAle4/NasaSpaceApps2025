import pandas as pd
import parameters as p

class PredictionBatch():
    def __init__(self):
        self.exoplanetDict = dict()
    
    def readCsvData(self, path: str):
        try:
            datafile = pd.read_csv(path, usecols=p.DATA_HEADERS)
            print('Data leida correctamente')
        except Exception as e:
            print(e)
        
class Exoplanet():
    def __init__(self):
        pass
    
if __name__ == '__main__':
    prediction_batch = PredictionBatch()
    prediction_batch.readCsvData("C:\\Users\\ADMIN\\Desktop\\Vicente\\Sistemas coheteria\\NasaSpaceApps2025\\Modelo\\koi_exoplanets.csv")
    
    
