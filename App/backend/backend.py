import pandas as pd
import parameters as p
import os

class PredictionBatch():
    def __init__(self):
        self.batchDataFrame = None
        
    
    def readCsvData(self, path: str):
        path = os.path.abspath(path)
        
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            return
            
        try:
            datafile = pd.read_csv(path, usecols=p.DATA_HEADERS)
            rows, cols = datafile.shape
            
            if cols == len(p.DATA_HEADERS):
                print(f"Se han cargado {rows} potenciales exoplanetas")
                self.batchDataFrame = datafile
            else:
                print(f"Error: no se han encontrado los datos necesarios en el archivo csv. {cols}/{p.DATA_HEADERS} cargados")
                self.batchDataFrame = None
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.batchDataFrame = None
        
        
        
class Exoplanet():
    def __init__(self):
        pass
    
if __name__ == '__main__':
    prediction_batch = PredictionBatch()
    prediction_batch.readCsvData(r"C:\Users\ADMIN\Desktop\Vicente\Sistemas coheteria\NasaSpaceApps\code\NasaSpaceApps2025\Modelo\koi_exoplanets.csv")
    
    #hola
    
