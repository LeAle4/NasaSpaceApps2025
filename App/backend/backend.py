import pandas as pd
import parameters as p
import os

class CurrentSesion():
    def __init__(self):
        self.currentBatches = dict()
        self.database = None
    
    def newPredictionBatch(self, path):
        new_batch = PredictionBatch()
        self.currentBatches[new_batch.id] = new_batch
    
    def init_database(self):
        # Inicializa la base de datos.
        database = Database()
        database.loadAllDatabase()
        self.database = database
    
    def addBatchToDatabase(self, batch_id: int):
        # Agrega un batch a la a la base de datos.
        pass

class PredictionBatch():
    _id_counter = 0
    def __init__(self):
        PredictionBatch._id_counter += 1
        self.id = PredictionBatch.id_counter 
        self.batchDataFrame = None
        self.confirmedExoplanets = None
        self.rejectedExoplanets = None
        
    
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
    
    def predictBatch(self):
        #Este metodo es ejecutado despues de cargar los datos. Ejecuta el modelo de predicción de exoplanetas y devuelve su veredicto.
        #Retorna: exoplanetas confirmados, exoplanetas rechazados.
        pass
        
        
        
class Database():
    # Esta clase contiene la data de todas las predicciones que se han hecho. Su información se guardará en un archivo
    # Y se cargará automaticamente cada vez que se inicia el programa.
    # Cada vez que se ejecuta una nuevo lote de predicciones (PredictionBatch) existirá la posibilidad de añadirlo a la 
    # base de datos general. Esta base de datos general se almacenará en un archivo de modo que su información no se pierda
    # al reiniciar el programa 
    def __init__(self):
        self.allData = None
    
    def loadAllDatabase(self):
        pass
    
    def addBatchToDatabase(self, batch_id: int):
        pass
    
if __name__ == '__main__':
    prediction_batch = PredictionBatch()
    prediction_batch.readCsvData(r"C:\Users\ADMIN\Desktop\Vicente\Sistemas coheteria\NasaSpaceApps\code\NasaSpaceApps2025\Modelo\koi_exoplanets.csv")
    
    #hola
    
