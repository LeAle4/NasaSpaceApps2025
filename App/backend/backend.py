import pandas as pd
import os
from PyQt5.QtCore import pyqtSignal, QObject
from . import parameters as p

class CurrentSesion(QObject):
    popup_msg_signal = pyqtSignal(str, str)
    batch_info_signal = pyqtSignal(dict)
    batch_data_signal = pyqtSignal(pd.DataFrame, int)  # Nueva señal para enviar datos del batch
    
    def __init__(self):
        super().__init__()
        self.currentBatches = dict()
        self.database = None
        self.init_database()
    
    def newPredictionBatch(self, path):
        new_batch = PredictionBatch()
        self.currentBatches[new_batch.id] = new_batch
        notification = new_batch.readCsvData(path)
        self.popup_msg_signal.emit(notification[0], notification[1])
        if notification[0] == "success":
            self.batch_info_signal.emit({
                "batch_id": new_batch.id,
                "batch_length": new_batch.batch_length,
                "confirmed": len(new_batch.confirmedExoplanets),
                "rejected": len(new_batch.rejectedExoplanets)
            })
    
    def clearBatches(self):
        self.currentBatches.clear()
        PredictionBatch._id_counter = 0 
    
    def removeBatch(self, batch_id: int):
        if batch_id in self.currentBatches:
            del self.currentBatches[batch_id]
            PredictionBatch._id_counter -= 1 
    
    def init_database(self):
        # Inicializa la base de datos.
        database = Database()
        database.loadAllDatabase()
        self.database = database
    
    def addBatchToDatabase(self, batch_id: int):
        # Agrega un batch a la a la base de datos.
        result = self.database.addBatchToDatabase(self.currentBatches[batch_id])
        self.popup_msg_signal.emit(result[0], result[1])
    
    def getBatchData(self, batch_id: int):
        """Envía los datos del batch solicitado al frontend"""
        if batch_id in self.currentBatches:
            batch = self.currentBatches[batch_id]
            if batch.batchDataFrame is not None:
                self.batch_data_signal.emit(batch.batchDataFrame, batch_id)
            else:
                # Enviar DataFrame vacío si no hay datos
                self.batch_data_signal.emit(pd.DataFrame(), batch_id)

class PredictionBatch():
    _id_counter = 0
    def __init__(self):
        PredictionBatch._id_counter += 1
        self.id = PredictionBatch._id_counter
        self.batch_length = 0
        self.batchDataFrame = None
        self.confirmedExoplanets = pd.DataFrame()
        self.rejectedExoplanets = pd.DataFrame()
        
    
    def readCsvData(self, path: str):
        path = os.path.abspath(path)
        
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            self.batchDataFrame = None
            return ["error", f"Error: File not found at {path}"]
            
        try:
            datafile = pd.read_csv(path, usecols=p.DATA_HEADERS)
            rows, cols = datafile.shape
            self.batch_length = rows
            
            if cols == len(p.DATA_HEADERS):
                print(f"Se han cargado {rows} potenciales exoplanetas")
                self.batchDataFrame = datafile
                return ["success", f"Se han cargado {rows} potenciales exoplanetas"]
            else:
                print(f"Error: no se han encontrado los datos necesarios en el archivo csv. {cols}/{p.DATA_HEADERS} cargados")
                
                self.batchDataFrame = None
                return ["error", f"Error: no se han encontrado los datos necesarios en el archivo csv. {cols}/{p.DATA_HEADERS} cargados"]
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.batchDataFrame = None
            return ["error", f"Error loading CSV: {e}"]
    
    def predictBatch(self):
        #Este metodo es ejecutado despues de cargar los datos. Ejecuta el modelo de predicción de exoplanetas y devuelve su veredicto.
        #Retorna: exoplanetas confirmados, exoplanetas rechazados.
        if self.batchDataFrame is None:
            print("Error: No hay datos cargados. Ejecuta readCsvData() primero.")
            return ["error", "Error: No hay datos cargados. Ejecuta readCsvData() primero."]
        
        
        
        
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
            return ["error", "Error: El batch no tiene datos de predicción. Ejecuta predictBatch() primero."]
        
        try:
            confirmed_batch = batch.confirmedExoplanets
            rejected_batch = batch.rejectedExoplanets
            
            # Concatenar con la base de datos existente
            
            self.allConfirmedExoplanets = pd.concat([self.allConfirmedExoplanets, confirmed_batch], ignore_index=True).drop_duplicates().reset_index(drop=True)
            self.allRejectedExoplanets = pd.concat([self.allRejectedExoplanets, rejected_batch], ignore_index=True).drop_duplicates().reset_index(drop=True)
            
            # Guardar la base de datos actualizada
            if self._saveDatabase():
                print(f"Batch {batch.id} añadido a la base de datos: {len(confirmed_batch)} confirmados, {len(rejected_batch)} rechazados")
                return ["success", f"Batch {batch.id} añadido a la base de datos: {len(confirmed_batch)} confirmados, {len(rejected_batch)} rechazados"]
            
        except Exception as e:
            print(f"Error añadiendo batch a la base de datos: {e}")
            return ["error", f"Error añadiendo batch a la base de datos: {e}"]

    def _saveDatabase(self):
        """Guarda toda la base de datos en un archivo CSV"""
        try:
            self.allConfirmedExoplanets.to_csv(self.confirmed_file_path, index=False)
            self.allRejectedExoplanets.to_csv(self.rejected_file_path, index=False)
            print(f"Base de datos guardada en {self.confirmed_file_path} y {self.rejected_file_path}")
            return ["success", f"Base de datos guardada en {self.confirmed_file_path} y {self.rejected_file_path}"]
            
        except Exception as e:
            print(f"Error guardando base de datos: {e}")
            return ["error", f"Error guardando base de datos: {e}"]

    def getDatabaseStats(self):
        """Obtiene estadísticas de la base de datos"""
        pass
    
if __name__ == '__main__':
    prediction_batch = PredictionBatch()
    prediction_batch.readCsvData(r"C:\Users\ADMIN\Desktop\Vicente\Sistemas coheteria\NasaSpaceApps\code\NasaSpaceApps2025\Modelo\koi_exoplanets.csv")
    
