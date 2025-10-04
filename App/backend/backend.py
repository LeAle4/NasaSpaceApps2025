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
        self.database.addBatchToDatabase(self, self.currentBatches[batch_id])

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
            self.batchDataFrame = None
            return False
            
        try:
            datafile = pd.read_csv(path, usecols=p.DATA_HEADERS)
            rows, cols = datafile.shape
            
            if cols == len(p.DATA_HEADERS):
                print(f"Se han cargado {rows} potenciales exoplanetas")
                self.batchDataFrame = datafile
                return True
            else:
                print(f"Error: no se han encontrado los datos necesarios en el archivo csv. {cols}/{p.DATA_HEADERS} cargados")
                self.batchDataFrame = None
                return False
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.batchDataFrame = None
            return False
    
    def predictBatch(self):
        #Este metodo es ejecutado despues de cargar los datos. Ejecuta el modelo de predicción de exoplanetas y devuelve su veredicto.
        #Retorna: exoplanetas confirmados, exoplanetas rechazados.
        pass
        
        
        
class Database():
    def __init__(self):
        self.allConfirmedExoplanets = pd.DataFrame()
        self.allRejectedExoplanets = pd.DataFrame()
        self.confirmed_file_path = "confirmed_exoplanets_data.csv"  # Archivo para persistencia
        self.rejected_file_path = "rejected_exoplanets_data.csv"
        
    def loadAllDatabase(self):
        """Carga la base de datos desde archivo si existe"""
        try:
            if os.path.exists(self.confirmed_file_path) and os.path.exists(self.rejected_file_path):
                self.allConfirmedExoplanets = pd.read_csv(self.confirmed_file_path)
                print(f"Se cargaron {len(self.allConfirmedExoplanets)} confirmados")
                self.allRejectedExoplanets = pd.read_csv(self.rejected_file_path)
                print(f"Se cargaron {len(self.allRejectedExoplanets)} rechazados")
                return True
            else:
                print("No se encontró archivo de base de datos. Se creará uno nuevo.")
                empty_confirmed = pd.DataFrame()
                empty_rejected = pd.DataFrame()
                try:
                    empty_confirmed.to_csv(self.confirmed_file_path, index=False)
                    empty_rejected.to_csv(self.rejected_file_path, index=False)
                    print(f"Se ha creado una base de datos en: {self.confirmed_file_path}")
                    print(f"Se ha creado una base de datos en: {self.rejected_file_path}")
                    return True
                except Exception as e:
                    print(f"Error creando archivos de base de datos: {e}")
                    return False
                
        except Exception as e:
            print(f"Error cargando base de datos: {e}")
            return False

    def addBatchToDatabase(self, batch: PredictionBatch):
        """Añade un batch de predicción a la base de datos"""
        if batch.confirmedExoplanets is None or batch.rejectedExoplanets is None:
            print("Error: El batch no tiene datos de predicción. Ejecuta predictBatch() primero.")
            return False
        
        try:
            confirmed_batch = batch.confirmedExoplanets
            rejected_batch = batch.rejectedExoplanets
            
            # Concatenar con la base de datos existente
            
            self.allConfirmedExoplanets = pd.concat([self.allConfirmedExoplanets, confirmed_batch], ignore_index=True)
            self.allRejectedExoplanets = pd.concat([self.allRejectedExoplanets, rejected_batch], ignore_index=True)
            
            # Guardar la base de datos actualizada
            if self._saveDatabase():
                print(f"Batch {batch.id} añadido a la base de datos: {len(confirmed_batch)} confirmados, {len(rejected_batch)} rechazados")
                return True
            
        except Exception as e:
            print(f"Error añadiendo batch a la base de datos: {e}")
            return False

    def _saveDatabase(self):
        """Guarda toda la base de datos en un archivo CSV"""
        try:
            self.allConfirmedExoplanets.to_csv(self.confirmed_file_path, index=False)
            self.allRejectedExoplanets.to_csv(self.rejected_file_path, index=False)
            print(f"Base de datos guardada en {self.confirmed_file_path} y {self.rejected_file_path}")
            return True
            
        except Exception as e:
            print(f"Error guardando base de datos: {e}")
            return False

    def getDatabaseStats(self):
        """Obtiene estadísticas de la base de datos"""
        pass
    
if __name__ == '__main__':
    prediction_batch = PredictionBatch()
    prediction_batch.readCsvData(r"C:\Users\ADMIN\Desktop\Vicente\Sistemas coheteria\NasaSpaceApps\code\NasaSpaceApps2025\Modelo\koi_exoplanets.csv")
    
    #hola
    
