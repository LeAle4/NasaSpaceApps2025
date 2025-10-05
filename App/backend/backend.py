import pandas as pd
import os
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from . import parameters as p
from .analysis import model as model_wrapper
from . import ml as ml_core

class CurrentSesion(QObject):
    """Application session manager for batches, background prediction, and DB.

    This QObject exposes signals used by the frontend and keeps session state
    such as the set of loaded `PredictionBatch` objects, the simple CSV-backed
    database and any background prediction threads.
    """
    popup_msg_signal = pyqtSignal(str, str)
    batch_info_signal = pyqtSignal(dict)
    prediction_progress_signal = pyqtSignal(int, str, str)  # batch_id, status, message
    batch_data_signal = pyqtSignal(pd.DataFrame, int)  # Signal to send batch data to frontend
    database_data_signal = pyqtSignal(pd.DataFrame, str)  # Signal to send DB data to frontend
    training_finished_signal = pyqtSignal(dict)  # emit dict with training result
    # Ask the frontend to show a Save File dialog. Payload: suggested_filename (str)
    request_model_save_signal = pyqtSignal(str)
    # Frontend replies with chosen path; payload: absolute_path (str)
    model_save_path_signal = pyqtSignal(str)

    def __init__(self):
        """Initialize session state and load persisted database files.

        The constructor creates containers for batches and prediction threads
        and attempts to initialise a persistent CSV-backed database.
        """
        super().__init__()
        self.currentBatches = dict()
        self.database = None
        # default model path (applies when a batch doesn't have its own model_path)
        self.default_model_path = None
        # track running prediction threads
        self._predict_threads = {}
        self.init_database()
        # storage for the most recently trained model wrapper (not persisted)
        self._last_trained_wrapper = None
        # connect incoming save-path replies to handler
        self.model_save_path_signal.connect(self._on_model_save_path)

    class BatchPredictThread(QThread):
        """Background thread wrapper to execute PredictionBatch.predictBatch.

        The thread catches exceptions and emits a `finished_signal(status, message, batch_id)`
        tuple so the UI can react without blocking.
        """
        finished_signal = pyqtSignal(str, str, int)  # status, message, batch_id

        def __init__(self, batch: 'PredictionBatch'):
            """Store the PredictionBatch instance to be executed on run()."""
            super().__init__()
            self.batch = batch

        def run(self):
            """Execute the batch prediction and emit completion status.

            The method expects `PredictionBatch.predictBatch` to return a list or
            tuple where the first two elements are (status, message). Any
            unexpected exception is forwarded as an error status.
            """
            try:
                res = self.batch.predictBatch()
                status, message = (res[0], res[1]) if isinstance(res, (list, tuple)) and len(res) >= 2 else ("error", "Unknown response")
                self.finished_signal.emit(status, message, self.batch.id)
            except Exception as e:
                # Convert unexpected exceptions into an error status for the UI
                self.finished_signal.emit("error", str(e), self.batch.id)

    class TrainingThread(QThread):
        """Run a training job in background using ml_core.train_from_database."""
        finished_signal = pyqtSignal(dict)

        def __init__(self, confirmed_csv: str, rejected_csv: str, out_dir: str, params: dict):
            super().__init__()
            self.confirmed_csv = confirmed_csv
            self.rejected_csv = rejected_csv
            self.out_dir = out_dir
            self.params = params or {}

        def run(self):
            try:
                # delegate to ml_core
                res = ml_core.train_from_database(
                    confirmed_csv=self.confirmed_csv,
                    rejected_csv=self.rejected_csv,
                    data_headers=p.DATA_HEADERS,
                    out_dir=self.out_dir,
                    params=self.params,
                )
                # Expect res to be a dict with at least status/message
                self.finished_signal.emit(res)
            except Exception as e:
                self.finished_signal.emit({'status': 'error', 'message': str(e), 'saved_paths': []})

    def startPrediction(self, batch_id: int):
        """Start prediction for a specific batch id in a background thread."""
        if batch_id not in self.currentBatches:
            self.popup_msg_signal.emit("error", f"Batch {batch_id} not found")
            return

        batch = self.currentBatches[batch_id]
        # emit started progress so UI can show indicator
        self.prediction_progress_signal.emit(batch_id, 'started', '')

        thread = CurrentSesion.BatchPredictThread(batch)

        def _on_done(status, message, bid):
            # emit progress update
            prog_status = 'completed' if status == 'success' else 'error'
            self.prediction_progress_signal.emit(bid, prog_status, message)

            # forward popup message
            self.popup_msg_signal.emit(status, message)
            # if success, update batch info in UI
            if status == "success":
                self.batch_info_signal.emit({
                    "batch_id": bid,
                    "batch_length": batch.batch_length,
                    "confirmed": len(batch.confirmedExoplanets) if batch.confirmedExoplanets is not None else 0,
                    "rejected": len(batch.rejectedExoplanets) if batch.rejectedExoplanets is not None else 0
                })
            # cleanup
            try:
                del self._predict_threads[bid]
            except KeyError:
                pass

        thread.finished_signal.connect(_on_done)
        self._predict_threads[batch_id] = thread
        thread.start()

    def startTraining(self, confirmed_csv: str, rejected_csv: str, out_dir: str, params: dict):
        """Start background training job and emit training_finished_signal on completion."""
        # Basic file checks (fail fast)
        if not os.path.exists(confirmed_csv):
            self.popup_msg_signal.emit('error', f'Confirmed file not found: {confirmed_csv}')
            return
        if not os.path.exists(rejected_csv):
            self.popup_msg_signal.emit('error', f'Rejected file not found: {rejected_csv}')
            return

        # Run training step-by-step synchronously so data is created in-order
        try:
            result = ml_core.train_from_database(
                confirmed_csv=confirmed_csv,
                rejected_csv=rejected_csv,
                data_headers=p.DATA_HEADERS,
                out_dir=out_dir or '',
                params=params or {},
            )
            # If training returned a model wrapper, keep it for potential save
            try:
                wrapper = result.get('model_wrapper')
                if wrapper is not None:
                    self._last_trained_wrapper = wrapper
            except Exception:
                pass
            status = result.get('status', 'success')
            message = result.get('message', 'Training completed')
            # notify UI
            self.popup_msg_signal.emit(status, message)
            # Ask frontend to prompt for model save if we have a wrapper
            if self._last_trained_wrapper is not None:
                # suggest a filename using out_dir if provided
                suggested = 'model.joblib'
                try:
                    if out_dir and os.path.isdir(out_dir):
                        suggested = os.path.join(out_dir, suggested)
                except Exception:
                    pass
                try:
                    self.request_model_save_signal.emit(suggested)
                except Exception:
                    pass
            try:
                self.training_finished_signal.emit(result)
            except Exception:
                pass
        except Exception as e:
            msg = str(e)
            self.popup_msg_signal.emit('error', msg)
            try:
                self.training_finished_signal.emit({'status': 'error', 'message': msg, 'saved_paths': []})
            except Exception:
                pass
    def _on_model_save_path(self, absolute_path: str):
        """Called when the frontend returns a save path for the last trained model.

        If we have a trained wrapper stored, persist it using Model.save and
        notify the frontend via popup_msg_signal. The method is defensive and
        emits an error if no trained model is available.
        """
        if not absolute_path:
            self.popup_msg_signal.emit('warning', 'No path provided for model save')
            return

        wrapper = getattr(self, '_last_trained_wrapper', None)
        if wrapper is None:
            self.popup_msg_signal.emit('error', 'No trained model available to save')
            return

        try:
            # ensure parent dir exists
            parent = os.path.dirname(absolute_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            # Model wrapper exposes save(path)
            wrapper.save(absolute_path)
            self.popup_msg_signal.emit('success', f'Model saved to: {absolute_path}')
            # clear stored wrapper after saving to avoid accidental reuse
            self._last_trained_wrapper = None
        except Exception as e:
            self.popup_msg_signal.emit('error', f'Failed to save model: {e}')
    
    def newPredictionBatch(self, path):
        """Create and register a PredictionBatch from a CSV path.

        Emits a popup with the load status and a `batch_info_signal` on success so
        the UI can display basic metadata (id, length, counts).
        """
        new_batch = PredictionBatch()
        self.currentBatches[new_batch.id] = new_batch
        notification = new_batch.readCsvData(path)
        # notification is [status, message]
        self.popup_msg_signal.emit(notification[0], notification[1])
        if notification[0] == "success":
            self.batch_info_signal.emit({
                "batch_id": new_batch.id,
                "batch_length": new_batch.batch_length,
                "confirmed": len(new_batch.confirmedExoplanets) if new_batch.confirmedExoplanets is not None else 0,
                "rejected": len(new_batch.rejectedExoplanets) if new_batch.rejectedExoplanets is not None else 0
            })
    
    def clearBatches(self):
        self.currentBatches.clear()
        PredictionBatch._id_counter = 0 
    
    def removeBatch(self, batch_id: int):
        if batch_id in self.currentBatches:
            del self.currentBatches[batch_id]
            PredictionBatch._id_counter -= 1 
    
    def init_database(self):
        # Initialize the CSV-backed database used by the session.
        database = Database()
        database.loadAllDatabase()
        self.database = database
    
    def addBatchToDatabase(self, batch_id: int):
        # Add a batch to the persistent database. Guard if database not available.
        if not self.database:
            self.popup_msg_signal.emit("error", "Database not initialized")
            return ["error", "Database not initialized"]

        result = self.database.addBatchToDatabase(self.currentBatches[batch_id])
        # result is expected to be [status, message]
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            self.popup_msg_signal.emit(result[0], result[1])
        else:
            # Defensive fallback
            self.popup_msg_signal.emit("error", "Failed to add batch to database")
    
    def getBatchData(self, batch_id: int):
        """Send the DataFrame for the requested batch to the frontend.

        If the batch has no DataFrame loaded the function emits an empty DataFrame
        so the frontend can handle the empty case uniformly.
        """
        if batch_id in self.currentBatches:
            batch = self.currentBatches[batch_id]
            if batch.batchDataFrame is not None:
                self.batch_data_signal.emit(batch.batchDataFrame, batch_id)
            else:
                # Send empty DataFrame if no data is available
                self.batch_data_signal.emit(pd.DataFrame(), batch_id)
    
    def getDatabaseData(self, db_type: str):
        """Send a copy of the requested database table to the frontend.

        db_type should be 'confirmed' or 'rejected'. If the database isn't
        initialized this function does nothing (caller should check availability).
        """
        if self.database:
            if db_type == "confirmed":
                self.database_data_signal.emit(self.database.allConfirmedExoplanets, "confirmed")
            elif db_type == "rejected":
                self.database_data_signal.emit(self.database.allRejectedExoplanets, "rejected")

    def setModelForBatch(self, model_path: str, batch_id: int = -1):
        """Assign a model path to a specific batch (by id), or to all batches when batch_id == -1.

        Performs a lightweight validation (file exists) and attempts to load via analysis.model.Model.load
        to provide early feedback. Emits popup_msg_signal with result.
        """
        # Resolve the provided path to an absolute path for validation
        resolved_model = os.path.abspath(model_path) if model_path else None

        if not resolved_model or not os.path.exists(resolved_model):
            self.popup_msg_signal.emit("error", f"Model file not found: {model_path}")
            return ["error", f"Model file not found: {model_path}"]

        # Try to load the model wrapper to validate the file is a readable model
        try:
            _ = model_wrapper.Model.load(resolved_model)
        except Exception as e:
            self.popup_msg_signal.emit("error", f"Failed to load model: {e}")
            return ["error", f"Failed to load model: {e}"]

        # If batch_id == -1 assign as default for the session
        if batch_id == -1:
            self.default_model_path = resolved_model
            # assign to all existing batches so they will use it unless overridden
            for bid, batch in self.currentBatches.items():
                setattr(batch, 'model_path', resolved_model)

            self.popup_msg_signal.emit("success", f"Default model set and assigned to {len(self.currentBatches)} batches: {model_path}")
            return ["success", f"Default model set and assigned to {len(self.currentBatches)} batches: {model_path}"]

        # assign to specific batch
        if batch_id in self.currentBatches:
            batch = self.currentBatches[batch_id]
            setattr(batch, 'model_path', resolved_model)
            self.popup_msg_signal.emit("success", f"Model assigned to Batch {batch_id}: {model_path}")
            return ["success", f"Model assigned to Batch {batch_id}: {model_path}"]
        else:
            self.popup_msg_signal.emit("error", f"Batch {batch_id} not found")
            return ["error", f"Batch {batch_id} not found"]

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
                # Successful load: set DataFrame and report success
                print(f"Loaded {rows} potential exoplanet candidates")
                self.batchDataFrame = datafile
                return ["success", f"Loaded {rows} potential exoplanet candidates"]
            else:
                # Column mismatch: provide a helpful diagnostic message
                print(f"Error: required columns not found in CSV. Columns read: {cols}/{len(p.DATA_HEADERS)}")

                self.batchDataFrame = None
                return ["error", f"Error: required columns not found in CSV. Columns read: {cols}/{len(p.DATA_HEADERS)}"]
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.batchDataFrame = None
            return ["error", f"Error loading CSV: {e}"]
    
    def predictBatch(self):
        #Este método es ejecutado después de cargar los datos. Ejecuta el modelo de predicción de exoplanetas y devuelve su veredicto.
        #Retorna: exoplanetas confirmados, exoplanetas rechazados.
        if self.batchDataFrame is None:
            # No data loaded: instruct caller to load data first
            print("Error: No data loaded. Call readCsvData() first.")
            return ["error", "Error: No data loaded. Call readCsvData() first."]
        # Delegate core prediction to ml.predict_batch while still allowing
        # callers (the GUI / CurrentSesion) to pass app-level parameters.
        try:
            # model_path can be set on the batch instance to override discovery
            model_path = getattr(self, 'model_path', None)
            res = ml_core.predict_batch(
                batch_df=self.batchDataFrame,
                data_headers=p.DATA_HEADERS,
                model_path=model_path,
                fillna_value=0,
                positive_class=1,
            )

            # assign results to batch attributes so the rest of the app works
            self.lastPredictionResults = {
                'model_path': res.get('model_path'),
                'predictions': res.get('predictions'),
                'probabilities': res.get('probabilities'),
                'results_df': res.get('results_df'),
            }
            self.confirmedExoplanets = res.get('confirmed_df')
            self.rejectedExoplanets = res.get('rejected_df')

            confirmed_count = len(self.confirmedExoplanets) if self.confirmedExoplanets is not None else 0
            rejected_count = len(self.rejectedExoplanets) if self.rejectedExoplanets is not None else 0
            print(f"Prediction complete: {confirmed_count} confirmed, {rejected_count} rejected")
            return ["success", f"Prediction complete: {confirmed_count} confirmed, {rejected_count} rejected"]

        except Exception as e:
            print(f"Error durante la predicción: {e}")
            return ["error", f"Error durante la predicción: {e}"]
        
        
class Database():
    def __init__(self):
        self.allConfirmedExoplanets = pd.DataFrame()
        self.allRejectedExoplanets = pd.DataFrame()
        self.confirmed_file_path = "confirmed_exoplanets_data.csv"  # Archivo para persistencia
        self.rejected_file_path = "rejected_exoplanets_data.csv"
        
    def loadAllDatabase(self):
        """Load persisted confirmed/rejected CSVs into memory.

        If both CSV files exist they are read into DataFrames. If they don't
        exist empty CSV files are created so subsequent saves succeed. Returns
        True on success and False on any IO error.
        """
        try:
            if os.path.exists(self.confirmed_file_path) and os.path.exists(self.rejected_file_path):
                self.allConfirmedExoplanets = pd.read_csv(self.confirmed_file_path)
                print(f"Loaded {len(self.allConfirmedExoplanets)} confirmed records")
                self.allRejectedExoplanets = pd.read_csv(self.rejected_file_path)
                print(f"Loaded {len(self.allRejectedExoplanets)} rejected records")
                return True
            else:
                print("Database files not found. Creating new empty database files.")
                empty_confirmed = pd.DataFrame()
                empty_rejected = pd.DataFrame()
                try:
                    empty_confirmed.to_csv(self.confirmed_file_path, index=False)
                    empty_rejected.to_csv(self.rejected_file_path, index=False)
                    print(f"Created new database files: {self.confirmed_file_path}")
                    print(f"Created new database files: {self.rejected_file_path}")
                    return True
                except Exception as e:
                    print(f"Error creating database files: {e}")
                    return False

        except Exception as e:
            print(f"Error loading database: {e}")
            return False

    def addBatchToDatabase(self, batch: PredictionBatch):
        """Append the batch's confirmed and rejected DataFrames to the DB.

        The method concatenates batch results to the in-memory DataFrames and
        writes them to disk via `_saveDatabase`. Returns [status, message].
        """
        if batch.confirmedExoplanets is None or batch.rejectedExoplanets is None:
            print("Error: The batch has no prediction results. Call predictBatch() first.")
            return ["error", "Error: The batch has no prediction results. Call predictBatch() first."]

        try:
            confirmed_batch = batch.confirmedExoplanets
            rejected_batch = batch.rejectedExoplanets

            # Concatenate with the existing database, drop duplicates and reset index
            self.allConfirmedExoplanets = pd.concat([self.allConfirmedExoplanets, confirmed_batch], ignore_index=True).drop_duplicates().reset_index(drop=True)
            self.allRejectedExoplanets = pd.concat([self.allRejectedExoplanets, rejected_batch], ignore_index=True).drop_duplicates().reset_index(drop=True)

            # Save the updated database to disk
            if self._saveDatabase():
                print(f"Batch {batch.id} added to database: {len(confirmed_batch)} confirmed, {len(rejected_batch)} rejected")
                return ["success", f"Batch {batch.id} added to database: {len(confirmed_batch)} confirmed, {len(rejected_batch)} rejected"]

        except Exception as e:
            print(f"Error adding batch to database: {e}")
            return ["error", f"Error adding batch to database: {e}"]

    def _saveDatabase(self):
        """Persist the in-memory confirmed/rejected tables to CSV files.

        Returns ['success', message] on success or ['error', message] on failure.
        """
        try:
            self.allConfirmedExoplanets.to_csv(self.confirmed_file_path, index=False)
            self.allRejectedExoplanets.to_csv(self.rejected_file_path, index=False)
            print(f"Database saved to {self.confirmed_file_path} and {self.rejected_file_path}")
            return ["success", f"Database saved to {self.confirmed_file_path} and {self.rejected_file_path}"]

        except Exception as e:
            print(f"Error saving database: {e}")
            return ["error", f"Error saving database: {e}"]

    def getDatabaseStats(self):
        """Return simple statistics about the in-memory database tables.

        Returns a dict with counts for confirmed and rejected records. This is a
        lightweight helper intended for UI summaries or telemetry.
        """
        try:
            return {
                'confirmed_count': int(len(self.allConfirmedExoplanets)) if self.allConfirmedExoplanets is not None else 0,
                'rejected_count': int(len(self.allRejectedExoplanets)) if self.allRejectedExoplanets is not None else 0,
            }
        except Exception as e:
            # On unexpected error, return zeroed stats and log the exception
            print(f"Error computing database stats: {e}")
            return {'confirmed_count': 0, 'rejected_count': 0}
    
if __name__ == '__main__':
    prediction_batch = PredictionBatch()
    prediction_batch.readCsvData(r"C:\Users\ADMIN\Desktop\Vicente\Sistemas coheteria\NasaSpaceApps\code\NasaSpaceApps2025\Modelo\koi_exoplanets.csv")