import sys
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import *
import pandas as pd
import os


class DataLoaderThread(QThread):
    """Background thread to load a DataFrame slice without blocking the UI.

    Emits `data_loaded(DataFrame, batch_id)` when the requested subset is ready.
    """
    data_loaded = pyqtSignal(pd.DataFrame, int)
    
    def __init__(self, dataframe, batch_id, start_row, end_row):
        super().__init__()
        self.dataframe = dataframe
        self.batch_id = batch_id
        self.start_row = start_row
        self.end_row = end_row
    
    def run(self):
        # Simula carga y emite los datos
        subset = self.dataframe.iloc[self.start_row:self.end_row]
        self.data_loaded.emit(subset, self.batch_id)


class DatabaseSortThread(QThread):
    """Background thread that sorts a DataFrame and emits the sorted result.

    The thread attempts numeric-aware sorting for object dtype columns to give
    natural numeric ordering when possible, falling back to pandas sort_values.
    """
    sort_completed = pyqtSignal(pd.DataFrame)
    
    def __init__(self, dataframe, column, ascending):
        super().__init__()
        self.dataframe = dataframe.copy()
        self.column = column
        self.ascending = ascending
    
    def run(self):
        try:
            # Intentar conversión numérica si es posible
            if self.dataframe[self.column].dtype == 'object':
                try:
                    temp_col = pd.to_numeric(self.dataframe[self.column], errors='coerce')
                    if not temp_col.isna().all():
                        sorted_indices = temp_col.argsort()
                        self.dataframe = self.dataframe.iloc[sorted_indices]
                        if not self.ascending:
                            self.dataframe = self.dataframe.iloc[::-1]
                    else:
                        self.dataframe = self.dataframe.sort_values(by=self.column, ascending=self.ascending)
                except:
                    self.dataframe = self.dataframe.sort_values(by=self.column, ascending=self.ascending)
            else:
                self.dataframe = self.dataframe.sort_values(by=self.column, ascending=self.ascending)
            
            self.dataframe = self.dataframe.reset_index(drop=True)
            self.sort_completed.emit(self.dataframe)
        except Exception as e:
            print(f"Error in sort thread: {e}")
            self.sort_completed.emit(self.dataframe)


class DatabaseSearchThread(QThread):
    """Background thread that filters a DataFrame by substring and optionally sorts it.

    Emits `search_completed(DataFrame)` when the filtered (and possibly sorted)
    DataFrame is ready.
    """
    search_completed = pyqtSignal(pd.DataFrame)
    
    def __init__(self, dataframe, column, search_text, sort_column=None, ascending=True):
        super().__init__()
        self.dataframe = dataframe.copy()
        self.column = column
        self.search_text = search_text
        self.sort_column = sort_column
        self.ascending = ascending
    
    def run(self):
        try:
            if self.search_text.strip() == "":
                result = self.dataframe
            else:
                # Búsqueda case-insensitive
                mask = self.dataframe[self.column].astype(str).str.contains(
                    self.search_text, 
                    case=False, 
                    na=False
                )
                result = self.dataframe[mask].copy()
            
            # Aplicar ordenamiento si existe
            if self.sort_column and self.sort_column in result.columns:
                if result[self.sort_column].dtype == 'object':
                    try:
                        temp_col = pd.to_numeric(result[self.sort_column], errors='coerce')
                        if not temp_col.isna().all():
                            sorted_indices = temp_col.argsort()
                            result = result.iloc[sorted_indices]
                            if not self.ascending:
                                result = result.iloc[::-1]
                        else:
                            result = result.sort_values(by=self.sort_column, ascending=self.ascending)
                    except:
                        result = result.sort_values(by=self.sort_column, ascending=self.ascending)
                else:
                    result = result.sort_values(by=self.sort_column, ascending=self.ascending)
            
            result = result.reset_index(drop=True)
            self.search_completed.emit(result)
        except Exception as e:
            print(f"Error in search thread: {e}")
            self.search_completed.emit(self.dataframe)


class MainWindow(QMainWindow):
    load_csv_signal = pyqtSignal(str)
    remove_batch_signal = pyqtSignal(int)
    clear_batches_signal = pyqtSignal()
    start_prediction_signal = pyqtSignal(int)
    load_model_signal = pyqtSignal(str, int)  # model_path, batch_id (-1 for all)
    save_to_database_signal = pyqtSignal(int)
    request_batch_data_signal = pyqtSignal(int)
    request_database_signal = pyqtSignal(str)  # Nueva señal para solicitar datos de la DB
    start_training_signal = pyqtSignal(str, str, str)  # confirmed_csv, rejected_csv, out_dir
    
    def __init__(self):
        super().__init__()
        self.current_dataframe = None
        self.current_batch_id = None
        self.current_page = 0
        self.rows_per_page = 500
        self.loader_thread = None
        
        # Variables para la pestaña de datos
        self.database_dataframe = None
        self.filtered_dataframe = None
        self.current_db_page = 0
        self.current_sort_column = None
        self.current_sort_order = Qt.AscendingOrder
        
        # Threads para operaciones de base de datos
        self.sort_thread = None
        self.search_thread = None
        
        # Debounce timer para la búsqueda
        self.search_timer = QtCore.QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.perform_search)
        
        self.setupUi()
        
    def setupUi(self):
        self.setObjectName("HopeFinder")
        self.resize(800, 600)
        
        # Widget central
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        
        # Layout principal
        self.mainLayout = QVBoxLayout(self.centralwidget)
        self.mainLayout.setContentsMargins(10, 10, 10, 10)
        
        # QTabWidget para las pestañas
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        
        # ========== PESTAÑA DE MODELOS ==========
        self.tabModelos = QWidget()
        self.tabModelos.setObjectName("tabModelos")
        self.modelosLayout = QVBoxLayout(self.tabModelos)
        self.hbox = QHBoxLayout()
        
        # Label principal
        self.labelModelos = QLabel("Import exoplanet data:")
        self.labelModelos.setFont(QFont("Arial", 12)) 
        self.labelModelos.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.modelosLayout.addWidget(self.labelModelos)
        
        # Frame para la lista de archivos
        self.batch_files_frame = QFrame()
        self.batch_files_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.batch_files_frame.setLineWidth(2)
        self.batch_files_frame.setMinimumHeight(200)
        
        # Layout para el frame
        self.frame_layout = QVBoxLayout(self.batch_files_frame)
        
        # Label para el frame
        self.label_archivos = QLabel("Loaded Data Batches:")
        self.label_archivos.setFont(QFont("Arial", 10, QFont.Bold))
        self.frame_layout.addWidget(self.label_archivos)
        
        # ListWidget para mostrar archivos
        self.lista_archivos = QListWidget()
        self.lista_archivos.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lista_archivos.itemSelectionChanged.connect(self.enable_button_selection)
        self.lista_archivos.itemSelectionChanged.connect(self.on_batch_selected)
        self.frame_layout.addWidget(self.lista_archivos)
        
        # Botones para gestionar archivos
        self.buttons_layout = QHBoxLayout()
        
        self.btn_start_prediction = QPushButton('Start Prediction')
        self.btn_start_prediction.clicked.connect(self.start_prediction)
        self.btn_start_prediction.setEnabled(False)

        self.btn_load_model = QPushButton('Load Model')
        self.btn_load_model.clicked.connect(self.load_model_clicked)
        self.btn_load_model.setEnabled(False)

        self.btn_abrir_csv = QPushButton('Add Data Batch')
        self.btn_abrir_csv.clicked.connect(self.open_csv_batch)
        
        self.btn_remove = QPushButton('Remove Selected')
        self.btn_remove.clicked.connect(self.remover_archivo)
        self.btn_remove.setEnabled(False)
        
        self.btn_clear = QPushButton('Clear All')
        self.btn_clear.clicked.connect(self.limpiar_archivos)
        self.btn_clear.setEnabled(False)
        
        self.btn_save_to_db = QPushButton('Save to Database')
        self.btn_save_to_db.clicked.connect(self.save_to_database)
        self.btn_save_to_db.setEnabled(False)
        
        self.buttons_layout.addWidget(self.btn_start_prediction)
        self.buttons_layout.addWidget(self.btn_load_model)
        self.buttons_layout.addWidget(self.btn_abrir_csv)
        self.buttons_layout.addWidget(self.btn_remove)
        self.buttons_layout.addWidget(self.btn_clear)
        self.buttons_layout.addWidget(self.btn_save_to_db)
        self.buttons_layout.addStretch()
        
        self.frame_layout.addLayout(self.buttons_layout)
        
        self.modelosLayout.addWidget(self.batch_files_frame)

        # ===== Training from database controls =====
        self.train_frame = QFrame()
        self.train_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.train_frame.setLineWidth(1)
        self.train_layout = QHBoxLayout(self.train_frame)

        self.label_train = QLabel("Train model from database:")
        self.train_layout.addWidget(self.label_train)

        self.btn_select_confirmed = QPushButton('Select Confirmed CSV')
        self.btn_select_confirmed.clicked.connect(self.select_confirmed_csv)
        self.train_layout.addWidget(self.btn_select_confirmed)

        self.btn_select_rejected = QPushButton('Select Rejected CSV')
        self.btn_select_rejected.clicked.connect(self.select_rejected_csv)
        self.train_layout.addWidget(self.btn_select_rejected)

        self.btn_select_outdir = QPushButton('Output Folder')
        self.btn_select_outdir.clicked.connect(self.select_outdir)
        self.train_layout.addWidget(self.btn_select_outdir)

        self.btn_start_training = QPushButton('Start Training')
        self.btn_start_training.clicked.connect(self.start_training_clicked)
        self.train_layout.addWidget(self.btn_start_training)

        self.modelosLayout.addWidget(self.train_frame)
        
        # Frame para mostrar datos del batch seleccionado
        self.data_viewer_frame = QFrame()
        self.data_viewer_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.data_viewer_frame.setLineWidth(2)
        self.data_viewer_frame.setMinimumHeight(200)
        
        # Layout para el frame de datos
        self.data_viewer_layout = QVBoxLayout(self.data_viewer_frame)
        
        # Header con label y controles de paginación
        self.header_layout = QHBoxLayout()
        
        # Label para el frame de datos
        self.label_data_viewer = QLabel("Exoplanet Data (Select a batch above):")
        self.label_data_viewer.setFont(QFont("Arial", 10, QFont.Bold))
        self.header_layout.addWidget(self.label_data_viewer)
        
        self.header_layout.addStretch()
        
        # Controles de paginación
        self.pagination_widget = QWidget()
        self.pagination_layout = QHBoxLayout(self.pagination_widget)
        self.pagination_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_first_page = QPushButton('<<')
        self.btn_first_page.setMaximumWidth(40)
        self.btn_first_page.clicked.connect(self.go_to_first_page)
        self.btn_first_page.setEnabled(False)
        
        self.btn_prev_page = QPushButton('<')
        self.btn_prev_page.setMaximumWidth(40)
        self.btn_prev_page.clicked.connect(self.go_to_prev_page)
        self.btn_prev_page.setEnabled(False)
        
        self.label_page_info = QLabel('Page 0 of 0')
        self.label_page_info.setAlignment(Qt.AlignCenter)
        self.label_page_info.setMinimumWidth(100)
        
        self.btn_next_page = QPushButton('>')
        self.btn_next_page.setMaximumWidth(40)
        self.btn_next_page.clicked.connect(self.go_to_next_page)
        self.btn_next_page.setEnabled(False)
        
        self.btn_last_page = QPushButton('>>')
        self.btn_last_page.setMaximumWidth(40)
        self.btn_last_page.clicked.connect(self.go_to_last_page)
        self.btn_last_page.setEnabled(False)
        
        self.pagination_layout.addWidget(self.btn_first_page)
        self.pagination_layout.addWidget(self.btn_prev_page)
        self.pagination_layout.addWidget(self.label_page_info)
        self.pagination_layout.addWidget(self.btn_next_page)
        self.pagination_layout.addWidget(self.btn_last_page)
        
        self.header_layout.addWidget(self.pagination_widget)
        self.pagination_widget.setVisible(False)
        
        self.data_viewer_layout.addLayout(self.header_layout)
        
        # Tabla para mostrar los datos
        self.table_exoplanets = QTableWidget()
        self.table_exoplanets.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_exoplanets.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_exoplanets.setAlternatingRowColors(True)
        self.data_viewer_layout.addWidget(self.table_exoplanets)
        
        self.modelosLayout.addWidget(self.data_viewer_frame)
        
        self.tabWidget.addTab(self.tabModelos, "Model")
        
        # ========== PESTAÑA DE DATOS (DATABASE) ==========
        self.tabDatos = QWidget()
        self.tabDatos.setObjectName("tabDatos")
        self.datosLayout = QVBoxLayout(self.tabDatos)
        
        # Label principal
        self.labelDatos = QLabel("Database Visualization:")
        self.labelDatos.setFont(QFont("Arial", 12))
        self.labelDatos.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.datosLayout.addWidget(self.labelDatos)
        
        # Frame para controles de la base de datos
        self.db_controls_frame = QFrame()
        self.db_controls_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.db_controls_frame.setLineWidth(2)
        self.db_controls_layout = QVBoxLayout(self.db_controls_frame)
        
        # Selector de base de datos
        self.db_selector_layout = QHBoxLayout()
        self.label_db_selector = QLabel("Select Database:")
        self.label_db_selector.setFont(QFont("Arial", 10, QFont.Bold))
        self.db_selector_layout.addWidget(self.label_db_selector)
        
        self.combo_db_selector = QComboBox()
        self.combo_db_selector.addItem("Confirmed Exoplanets")
        self.combo_db_selector.addItem("Rejected Exoplanets")
        self.combo_db_selector.currentTextChanged.connect(self.on_database_changed)
        self.db_selector_layout.addWidget(self.combo_db_selector)
        
        self.btn_refresh_db = QPushButton("Refresh")
        self.btn_refresh_db.clicked.connect(self.refresh_database)
        self.db_selector_layout.addWidget(self.btn_refresh_db)
        
        self.db_selector_layout.addStretch()
        self.db_controls_layout.addLayout(self.db_selector_layout)
        
        # Controles de ordenamiento y búsqueda
        self.search_sort_layout = QHBoxLayout()
        
        # Ordenamiento
        self.label_sort = QLabel("Sort by:")
        self.search_sort_layout.addWidget(self.label_sort)
        
        self.combo_sort_column = QComboBox()
        self.combo_sort_column.setMinimumWidth(150)
        self.search_sort_layout.addWidget(self.combo_sort_column)
        
        self.btn_sort_asc = QPushButton("↑ Ascending")
        self.btn_sort_asc.clicked.connect(lambda: self.sort_database(Qt.AscendingOrder))
        self.search_sort_layout.addWidget(self.btn_sort_asc)
        
        self.btn_sort_desc = QPushButton("↓ Descending")
        self.btn_sort_desc.clicked.connect(lambda: self.sort_database(Qt.DescendingOrder))
        self.search_sort_layout.addWidget(self.btn_sort_desc)
        
        self.search_sort_layout.addSpacing(20)
        
        # Búsqueda
        self.label_search = QLabel("Search:")
        self.search_sort_layout.addWidget(self.label_search)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search term...")
        self.search_input.setMinimumWidth(200)
        self.search_input.textChanged.connect(self.on_search_text_changed)
        self.search_sort_layout.addWidget(self.search_input)
        
        self.btn_clear_search = QPushButton("Clear")
        self.btn_clear_search.clicked.connect(self.clear_search)
        self.search_sort_layout.addWidget(self.btn_clear_search)
        
        # Indicador de carga
        self.label_loading = QLabel("⏳ Loading...")
        self.label_loading.setStyleSheet("color: #FF6B35; font-weight: bold;")
        self.label_loading.setVisible(False)
        self.search_sort_layout.addWidget(self.label_loading)
        
        self.search_sort_layout.addStretch()
        
        self.db_controls_layout.addLayout(self.search_sort_layout)
        
        self.datosLayout.addWidget(self.db_controls_frame)
        
        # Frame para mostrar datos de la base de datos
        self.db_viewer_frame = QFrame()
        self.db_viewer_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.db_viewer_frame.setLineWidth(2)
        self.db_viewer_layout = QVBoxLayout(self.db_viewer_frame)
        
        # Header con información y paginación
        self.db_header_layout = QHBoxLayout()
        
        self.label_db_info = QLabel("No data loaded")
        self.label_db_info.setFont(QFont("Arial", 10, QFont.Bold))
        self.db_header_layout.addWidget(self.label_db_info)
        
        self.db_header_layout.addStretch()
        
        # Controles de paginación para database
        self.db_pagination_widget = QWidget()
        self.db_pagination_layout = QHBoxLayout(self.db_pagination_widget)
        self.db_pagination_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_db_first = QPushButton('<<')
        self.btn_db_first.setMaximumWidth(40)
        self.btn_db_first.clicked.connect(self.go_to_first_db_page)
        self.btn_db_first.setEnabled(False)
        
        self.btn_db_prev = QPushButton('<')
        self.btn_db_prev.setMaximumWidth(40)
        self.btn_db_prev.clicked.connect(self.go_to_prev_db_page)
        self.btn_db_prev.setEnabled(False)
        
        self.label_db_page = QLabel('Page 0 of 0')
        self.label_db_page.setAlignment(Qt.AlignCenter)
        self.label_db_page.setMinimumWidth(100)
        
        self.btn_db_next = QPushButton('>')
        self.btn_db_next.setMaximumWidth(40)
        self.btn_db_next.clicked.connect(self.go_to_next_db_page)
        self.btn_db_next.setEnabled(False)
        
        self.btn_db_last = QPushButton('>>')
        self.btn_db_last.setMaximumWidth(40)
        self.btn_db_last.clicked.connect(self.go_to_last_db_page)
        self.btn_db_last.setEnabled(False)
        
        self.db_pagination_layout.addWidget(self.btn_db_first)
        self.db_pagination_layout.addWidget(self.btn_db_prev)
        self.db_pagination_layout.addWidget(self.label_db_page)
        self.db_pagination_layout.addWidget(self.btn_db_next)
        self.db_pagination_layout.addWidget(self.btn_db_last)
        
        self.db_header_layout.addWidget(self.db_pagination_widget)
        
        self.db_viewer_layout.addLayout(self.db_header_layout)
        
        # Tabla para mostrar datos de la base de datos
        self.table_database = QTableWidget()
        self.table_database.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_database.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_database.setAlternatingRowColors(True)
        self.table_database.setSortingEnabled(False)  # Desactivamos el sorting nativo
        self.db_viewer_layout.addWidget(self.table_database)
        
        self.datosLayout.addWidget(self.db_viewer_frame)
        
        self.tabWidget.addTab(self.tabDatos, "Data")
        
        # Agregar el TabWidget al layout principal
        self.mainLayout.addWidget(self.tabWidget)
        
        self.setCentralWidget(self.centralwidget)
        
        # Barra de estado
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        
        QtCore.QMetaObject.connectSlotsByName(self)
        
        # Cargar base de datos inicial
        self.refresh_database()

    # ========== MÉTODOS ORIGINALES (PESTAÑA MODELO) ==========
    
    def open_csv_batch(self):
        """Open a file dialog to select a CSV batch and perform basic validation.

        Emits `load_csv_signal(file_path)` when a valid CSV is selected. UI
        errors are shown with QMessageBox dialogs.
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Seleccionar archivo de exoplanetas",
                "",
                "Archivos CSV (*.csv);;Todos los archivos (*)"
            )
            
            if not file_path:
                return None
                
            if not file_path.lower().endswith('.csv'):
                QMessageBox.warning(
                    self, 
                    "Formato incorrecto", 
                    "Por favor selecciona un archivo CSV"
                )
                return None
                
            import os
            if not os.path.exists(file_path):
                QMessageBox.critical(
                    self,
                    "Archivo no encontrado",
                    f"El archivo no existe:\n{file_path}"
                )
                return None
                
            self.load_csv_signal.emit(file_path)
            
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error inesperado",
                f"Error al abrir el archivo:\n{str(e)}"
            )
            return None

    # ===== Training-related UI handlers =====
    def select_confirmed_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select confirmed CSV", "", "CSV Files (*.csv);;All files (*)")
        if file_path:
            self._confirmed_csv = file_path
            self.statusbar.showMessage(f"Confirmed CSV selected", 2000)

    def select_rejected_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select rejected CSV", "", "CSV Files (*.csv);;All files (*)")
        if file_path:
            self._rejected_csv = file_path
            self.statusbar.showMessage(f"Rejected CSV selected", 2000)

    def select_outdir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output directory", "")
        if folder:
            self._out_dir = folder
            self.statusbar.showMessage(f"Output folder set", 2000)

    def start_training_clicked(self):
        # Ensure paths exist
        confirmed = getattr(self, '_confirmed_csv', None)
        rejected = getattr(self, '_rejected_csv', None)
        outdir = getattr(self, '_out_dir', None)
        if not confirmed or not rejected:
            QMessageBox.warning(self, "Missing files", "Please select both confirmed and rejected CSV files before training.")
            return
        # Emit signal to backend
        self.start_training_signal.emit(confirmed, rejected, outdir or '')
        self.statusbar.showMessage("Training started in background", 2000)
    
    def add_batch_info(self, batch_info: dict):
        item_text = f"Batch {batch_info['batch_id']}.  Length {batch_info['batch_length']}  Confirmed {batch_info['confirmed']}  Rejected {batch_info['rejected']}"
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, batch_info["batch_id"])
        self.lista_archivos.addItem(item)
        self.btn_clear.setEnabled(True)
        self.btn_remove.setEnabled(True)
        self.statusbar.showMessage("Batch loaded successfully", 2000)
        self.enable_button_selection()
        
    def remover_archivo(self):
        """Remove the selected batch entry from the list and notify backend.

        This method updates the UI list widget and emits `remove_batch_signal`
        with the removed batch id so the backend can free related resources.
        """
        items_seleccionados = self.lista_archivos.selectedItems()
        if not items_seleccionados:
            return
        
        item = items_seleccionados[0]
        row = self.lista_archivos.row(item)
        self.lista_archivos.takeItem(row)
        self.statusbar.showMessage("Archivo removido", 2000)
        self.remove_batch_signal.emit(item.data(Qt.UserRole))
        
        self.clear_table_view()
    
    def limpiar_archivos(self):
        """Clear all loaded batches after user confirmation.

        Emits `clear_batches_signal` so the backend can reset session state.
        """
        reply = QMessageBox.question(
            self,
            "Confirmar",
            "¿Desea eliminar todos los archivos cargados?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.lista_archivos.clear()
            self.btn_clear.setEnabled(False)
            self.btn_start_prediction.setEnabled(False)
            self.statusbar.showMessage("Todos los archivos han sido removidos", 2000)
            self.clear_batches_signal.emit()
            
            self.clear_table_view()
    
    def clear_table_view(self):
        """Clear the current table view and reset pagination/UI state."""
        self.table_exoplanets.setRowCount(0)
        self.table_exoplanets.setColumnCount(0)
        self.label_data_viewer.setText("Exoplanet Data (Select a batch above):")
        self.pagination_widget.setVisible(False)
        self.current_dataframe = None
        self.current_batch_id = None
        self.current_page = 0
    
    def save_to_database(self):
        items_seleccionados = self.lista_archivos.selectedItems()
        if not items_seleccionados:
            return
        
        item = items_seleccionados[0]
        self.save_to_database_signal.emit(item.data(Qt.UserRole))
        
    def start_prediction(self):
        """Start prediction for the selected batch or for all loaded batches.

        Emits `start_prediction_signal(batch_id)` for each batch to predict. If
        no batch is selected the user is asked whether to run predictions on
        all loaded batches.
        """
        items_seleccionados = self.lista_archivos.selectedItems()
        if items_seleccionados:
            # Predict for selected (single-selection by UI config)
            item = items_seleccionados[0]
            batch_id = item.data(Qt.UserRole)
            self.start_prediction_signal.emit(batch_id)
            self.statusbar.showMessage(f"Prediction started for Batch {batch_id}", 2000)
            return

        # If nothing selected, offer to predict all loaded batches
        if self.lista_archivos.count() == 0:
            QMessageBox.information(self, "No batches", "No data batches loaded to predict.")
            return

        reply = QMessageBox.question(
            self,
            "Predict all?",
            "No batch selected. Do you want to run prediction on all loaded batches?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # iterate through all items and emit start signal for each
            for i in range(self.lista_archivos.count()):
                item = self.lista_archivos.item(i)
                batch_id = item.data(Qt.UserRole)
                self.start_prediction_signal.emit(batch_id)
            self.statusbar.showMessage("Prediction started for all loaded batches", 2000)

    def load_model_clicked(self):
        """Open a file dialog to choose a model (joblib). Ask whether to assign to selected batch or all."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model file",
            "",
            "Joblib files (*.joblib);;All files (*.*)"
        )

        if not file_path:
            return

        # If a batch is selected, offer to assign to selected batch or all
        items = self.lista_archivos.selectedItems()
        if items:
            item = items[0]
            batch_id = item.data(Qt.UserRole)
            reply = QMessageBox.question(
                self,
                "Assign model",
                f"Assign selected model to Batch {batch_id}?\nChoose No to set as default for all batches.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Yes:
                self.load_model_signal.emit(file_path, batch_id)
            else:
                # set as default for all
                self.load_model_signal.emit(file_path, -1)
        else:
            # no selection -> set as default for all batches
            self.load_model_signal.emit(file_path, -1)

    def show_msg(self, tipo, mensaje):
        if tipo == "success":
            QMessageBox.information(self, "Success!", mensaje)
        elif tipo == "error":
            QMessageBox.critical(self, "Error", mensaje)
        elif tipo == "warning":
            QMessageBox.warning(self, "Warning", mensaje)
        else:
            QMessageBox.information(self, "Information", mensaje)
    
    def enable_button_selection(self):
        """Enable or disable UI buttons depending on the current selection state.

        This keeps the Start Prediction / Save to DB / Load Model buttons in sync
        with the user's selection and available batches.
        """
        if self.lista_archivos.selectedItems():
            self.btn_start_prediction.setEnabled(True)
            self.btn_save_to_db.setEnabled(True)
            self.btn_load_model.setEnabled(True)
        else:
            self.btn_start_prediction.setEnabled(False)
            # enable load model if there are any batches
            self.btn_load_model.setEnabled(self.lista_archivos.count() > 0)
        if self.lista_archivos.count() == 0:
            self.btn_clear.setEnabled(False)
            self.btn_remove.setEnabled(False)
    
    def on_batch_selected(self):
        """Called when a batch is selected in the list; request its data from backend."""
        items_seleccionados = self.lista_archivos.selectedItems()
        if items_seleccionados:
            batch_id = items_seleccionados[0].data(Qt.UserRole)
            self.request_batch_data_signal.emit(batch_id)

    def on_prediction_progress(self, batch_id: int, status: str, message: str):
        """Handle backend prediction progress updates.

        status is one of: 'started', 'completed', 'error'. The method updates
        UI text and enables/disables controls appropriately.
        """
        # Find the list item for this batch
        item = None
        for i in range(self.lista_archivos.count()):
            it = self.lista_archivos.item(i)
            if it.data(Qt.UserRole) == batch_id:
                item = it
                break

        if item is None:
            return

        if status == 'started':
            item.setText(f"Batch {batch_id} - Processing...")
            # disable controls to avoid double-starting
            self.btn_start_prediction.setEnabled(False)
            self.btn_save_to_db.setEnabled(False)
        else:
            # Completed or error - restore the text via batch_info or show error
            if status == 'completed':
                # Request the backend to re-send batch info which will update the item
                # The existing flow will call add_batch_info when batch_info_signal is emitted
                # Here we optimistically re-enable controls
                self.btn_start_prediction.setEnabled(True)
                self.btn_save_to_db.setEnabled(True)
            elif status == 'error':
                # show a brief message in statusbar
                self.statusbar.showMessage(f"Prediction error for Batch {batch_id}: {message}", 5000)
                self.btn_start_prediction.setEnabled(True)
                self.btn_save_to_db.setEnabled(True)
    
    def display_batch_data(self, dataframe: pd.DataFrame, batch_id: int):
        """Display a batch DataFrame in the table widget with pagination.

        Accepts an empty DataFrame which will clear the table and show a friendly
        message. The method keeps the full DataFrame in memory and renders the
        current page into the QTableWidget.
        """
        if dataframe is None or dataframe.empty:
            self.clear_table_view()
            self.label_data_viewer.setText(f"Batch {batch_id}: No data available")
            return
        
        # Guardar referencia al dataframe completo
        self.current_dataframe = dataframe
        self.current_batch_id = batch_id
        self.current_page = 0
        
        # Mostrar controles de paginación solo si hay más de una página
        total_pages = (len(dataframe) + self.rows_per_page - 1) // self.rows_per_page
        if total_pages > 1:
            self.pagination_widget.setVisible(True)
        else:
            self.pagination_widget.setVisible(False)
        
        # Cargar primera página
        self.load_current_page()
    
    def load_current_page(self):
        """Render the current page of the active batch into the table widget."""
        if self.current_dataframe is None:
            return
        
        total_rows = len(self.current_dataframe)
        total_pages = (total_rows + self.rows_per_page - 1) // self.rows_per_page
        
        # Calcular rango de filas para la página actual
        start_row = self.current_page * self.rows_per_page
        end_row = min(start_row + self.rows_per_page, total_rows)
        
        # Obtener subset de datos
        page_data = self.current_dataframe.iloc[start_row:end_row]
        
        # Actualizar label
        self.label_data_viewer.setText(
            f"Batch {self.current_batch_id} - Exoplanet Data "
            f"(Showing {start_row+1}-{end_row} of {total_rows})"
        )
        
        # Actualizar info de página
        self.label_page_info.setText(f'Page {self.current_page + 1} of {total_pages}')
        
        # Configurar tabla
        self.table_exoplanets.setRowCount(len(page_data))
        self.table_exoplanets.setColumnCount(len(page_data.columns))
        self.table_exoplanets.setHorizontalHeaderLabels(page_data.columns.tolist())
        
        # Llenar tabla con datos
        for i in range(len(page_data)):
            for j, col in enumerate(page_data.columns):
                value = page_data.iloc[i, j]
                item = QTableWidgetItem(str(value))
                self.table_exoplanets.setItem(i, j, item)
        
        # Ajustar tamaño de columnas
        self.table_exoplanets.resizeColumnsToContents()
        
        # Actualizar estado de botones de paginación
        self.btn_first_page.setEnabled(self.current_page > 0)
        self.btn_prev_page.setEnabled(self.current_page > 0)
        self.btn_next_page.setEnabled(self.current_page < total_pages - 1)
        self.btn_last_page.setEnabled(self.current_page < total_pages - 1)
        
        self.statusbar.showMessage(
            f"Displaying rows {start_row+1}-{end_row} of {total_rows} from Batch {self.current_batch_id}", 
            3000
        )
    
    def go_to_first_page(self):
        self.current_page = 0
        self.load_current_page()
    
    def go_to_prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.load_current_page()
    
    def go_to_next_page(self):
        if self.current_dataframe is not None:
            total_pages = (len(self.current_dataframe) + self.rows_per_page - 1) // self.rows_per_page
            if self.current_page < total_pages - 1:
                self.current_page += 1
                self.load_current_page()
    
    def go_to_last_page(self):
        if self.current_dataframe is not None:
            total_pages = (len(self.current_dataframe) + self.rows_per_page - 1) // self.rows_per_page
            self.current_page = total_pages - 1
            self.load_current_page()

    # ========== MÉTODOS NUEVOS (PESTAÑA DATOS) ==========
    
    def refresh_database(self):
        """Request the backend to refresh the chosen database table and update UI."""
        db_type = "confirmed" if self.combo_db_selector.currentText() == "Confirmed Exoplanets" else "rejected"
        self.request_database_signal.emit(db_type)
        self.statusbar.showMessage("Refreshing database...", 1000)
    
    def on_database_changed(self):
        """Called when the database selector changes; triggers a refresh."""
        self.refresh_database()
    
    def display_database_data(self, dataframe: pd.DataFrame, db_type: str):
        """Render database DataFrame into the database table view.

        Handles the empty case and initializes sorting/search controls.
        """
        if dataframe is None or dataframe.empty:
            self.database_dataframe = None
            self.filtered_dataframe = None
            self.label_db_info.setText(f"No data in {db_type} database")
            self.table_database.setRowCount(0)
            self.table_database.setColumnCount(0)
            self.combo_sort_column.clear()
            return
        
        # Guardar dataframe completo
        self.database_dataframe = dataframe.copy()
        self.filtered_dataframe = dataframe.copy()
        self.current_db_page = 0
        self.current_sort_column = None
        self.current_sort_order = Qt.AscendingOrder
        
        # Actualizar combo de columnas para sorting
        self.combo_sort_column.clear()
        self.combo_sort_column.addItems(dataframe.columns.tolist())
        
        # Limpiar búsqueda
        self.search_input.clear()
        
        # Cargar datos
        self.load_current_db_page()
        
        self.statusbar.showMessage(f"Loaded {len(dataframe)} records from {db_type} database", 2000)
    
    def sort_database(self, order: Qt.SortOrder):
        """Sort the currently displayed database page using a background thread.

        Uses `DatabaseSortThread` to avoid blocking the UI when sorting large tables.
        """
        if self.filtered_dataframe is None or self.combo_sort_column.currentText() == "":
            return
        
        # Verificar si ya hay un thread corriendo
        if self.sort_thread and self.sort_thread.isRunning():
            return
        
        column = self.combo_sort_column.currentText()
        self.current_sort_column = column
        self.current_sort_order = order
        ascending = (order == Qt.AscendingOrder)
        
        # Mostrar indicador de carga
        self.label_loading.setVisible(True)
        self.disable_database_controls(True)
        
        # Crear y ejecutar thread de ordenamiento
        self.sort_thread = DatabaseSortThread(self.filtered_dataframe, column, ascending)
        self.sort_thread.sort_completed.connect(self.on_sort_completed)
        self.sort_thread.start()
    
    def on_sort_completed(self, sorted_dataframe):
        """Callback invoked when sorting completes; update UI and controls."""
        self.filtered_dataframe = sorted_dataframe
        self.current_db_page = 0
        self.load_current_db_page()
        
        # Ocultar indicador de carga
        self.label_loading.setVisible(False)
        self.disable_database_controls(False)
        
        order_text = "ascending" if self.current_sort_order == Qt.AscendingOrder else "descending"
        self.statusbar.showMessage(f"Sorted by {self.current_sort_column} ({order_text})", 2000)
    
    def on_search_text_changed(self):
        """Called when the search input changes; debounces and schedules a search."""
        # Detener el timer anterior y reiniciarlo
        self.search_timer.stop()
        self.search_timer.start(500)  # Espera 500ms después de que el usuario deje de escribir
    
    def perform_search(self):
        """Perform the search/filter operation in a background thread.

        The function preserves responsiveness by re-scheduling searches if a
        previous search is still running.
        """
        search_text = self.search_input.text().strip()
        
        if self.database_dataframe is None:
            return
        
        # Si ya hay un thread corriendo, esperar a que termine
        if self.search_thread and self.search_thread.isRunning():
            # En lugar de retornar, programar otra búsqueda
            self.search_timer.start(200)
            return
        
        if search_text != "" and self.combo_sort_column.currentText() == "":
            # No mostrar warning durante escritura, simplemente retornar
            return
        
        # Mostrar indicador de carga solo si no estamos escribiendo activamente
        # (el timer garantiza que el usuario ya dejó de escribir)
        self.label_loading.setVisible(True)
        
        # NO deshabilitar controles para permitir escritura fluida
        # self.disable_database_controls(True)
        
        column = self.combo_sort_column.currentText() if search_text != "" else None
        ascending = (self.current_sort_order == Qt.AscendingOrder)
        
        # Crear y ejecutar thread de búsqueda
        self.search_thread = DatabaseSearchThread(
            self.database_dataframe, 
            column if column else "", 
            search_text,
            self.current_sort_column,
            ascending
        )
        self.search_thread.search_completed.connect(self.on_search_completed)
        self.search_thread.start()
    
    def on_search_completed(self, result_dataframe):
        """Callback invoked when a search thread completes; update UI with results."""
        self.filtered_dataframe = result_dataframe
        self.current_db_page = 0
        self.load_current_db_page()
        
        # Ocultar indicador de carga
        self.label_loading.setVisible(False)
        # NO re-habilitar controles porque nunca los deshabilitamos
        # self.disable_database_controls(False)
        
        search_text = self.search_input.text().strip()
        if search_text:
            self.statusbar.showMessage(
                f"Found {len(self.filtered_dataframe)} results for '{search_text}'", 
                2000
            )
        else:
            self.statusbar.showMessage(f"Showing all {len(self.filtered_dataframe)} records", 2000)
    
    def disable_database_controls(self, disable):
        """Enable/disable database controls while background processing is running."""
        self.combo_db_selector.setEnabled(not disable)
        self.btn_refresh_db.setEnabled(not disable)
        self.combo_sort_column.setEnabled(not disable)
        self.btn_sort_asc.setEnabled(not disable)
        self.btn_sort_desc.setEnabled(not disable)
        # NO deshabilitar search_input y btn_clear_search para mantener fluidez
        # self.search_input.setEnabled(not disable)
        # self.btn_clear_search.setEnabled(not disable)
    
    def clear_search(self):
        """Clear the search input and trigger a refresh via the debounce timer."""
        self.search_input.clear()
        # El timer se encargará de ejecutar perform_search
    
    def load_current_db_page(self):
        """Carga la página actual de la base de datos"""
        if self.filtered_dataframe is None or self.filtered_dataframe.empty:
            self.table_database.setRowCount(0)
            self.table_database.setColumnCount(0)
            self.label_db_info.setText("No data to display")
            self.label_db_page.setText('Page 0 of 0')
            self.btn_db_first.setEnabled(False)
            self.btn_db_prev.setEnabled(False)
            self.btn_db_next.setEnabled(False)
            self.btn_db_last.setEnabled(False)
            return
        
        total_rows = len(self.filtered_dataframe)
        total_pages = (total_rows + self.rows_per_page - 1) // self.rows_per_page
        
        # Calcular rango de filas
        start_row = self.current_db_page * self.rows_per_page
        end_row = min(start_row + self.rows_per_page, total_rows)
        
        # Obtener subset de datos
        page_data = self.filtered_dataframe.iloc[start_row:end_row]
        
        # Actualizar label
        db_name = self.combo_db_selector.currentText()
        self.label_db_info.setText(
            f"{db_name} - Showing {start_row+1}-{end_row} of {total_rows} records"
        )
        
        # Actualizar info de página
        self.label_db_page.setText(f'Page {self.current_db_page + 1} of {total_pages}')
        
        # Configurar tabla
        self.table_database.setRowCount(len(page_data))
        self.table_database.setColumnCount(len(page_data.columns))
        self.table_database.setHorizontalHeaderLabels(page_data.columns.tolist())
        
        # Llenar tabla con datos
        for i in range(len(page_data)):
            for j, col in enumerate(page_data.columns):
                value = page_data.iloc[i, j]
                item = QTableWidgetItem(str(value))
                self.table_database.setItem(i, j, item)
        
        # Ajustar tamaño de columnas
        self.table_database.resizeColumnsToContents()
        
        # Actualizar estado de botones de paginación
        self.btn_db_first.setEnabled(self.current_db_page > 0)
        self.btn_db_prev.setEnabled(self.current_db_page > 0)
        self.btn_db_next.setEnabled(self.current_db_page < total_pages - 1)
        self.btn_db_last.setEnabled(self.current_db_page < total_pages - 1)
    
    def go_to_first_db_page(self):
        self.current_db_page = 0
        self.load_current_db_page()
    
    def go_to_prev_db_page(self):
        if self.current_db_page > 0:
            self.current_db_page -= 1
            self.load_current_db_page()
    
    def go_to_next_db_page(self):
        if self.filtered_dataframe is not None:
            total_pages = (len(self.filtered_dataframe) + self.rows_per_page - 1) // self.rows_per_page
            if self.current_db_page < total_pages - 1:
                self.current_db_page += 1
                self.load_current_db_page()
    
    def go_to_last_db_page(self):
        if self.filtered_dataframe is not None:
            total_pages = (len(self.filtered_dataframe) + self.rows_per_page - 1) // self.rows_per_page
            self.current_db_page = total_pages - 1
            self.load_current_db_page()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec_())