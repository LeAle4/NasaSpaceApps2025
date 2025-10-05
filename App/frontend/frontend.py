import sys
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import *
import pandas as pd
import os
from backend.analysis import dataio
import time


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
    start_training_signal = pyqtSignal(str, str, str, dict)  # confirmed_csv, rejected_csv, out_dir, params
    # frontend will emit the chosen save-path for the model back to backend
    model_save_path_chosen = pyqtSignal(str)
    
    
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
        
        # Store orbit visualization windows
        self.orbit_windows = []
        
        self.setup_window_properties()
        self.setupUi()
        
    def setupUi(self):
        self.setObjectName("HopeFinder")
        
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
        # Connect row selection to enable orbit button
        self.table_exoplanets.itemSelectionChanged.connect(self.on_exoplanet_row_selected)
        self.data_viewer_layout.addWidget(self.table_exoplanets)
        
        # Add orbit visualization button below table
        self.orbit_button_layout = QHBoxLayout()
        self.btn_visualize_orbit = QPushButton('🌍 Visualize Orbit')
        self.btn_visualize_orbit.setEnabled(False)
        self.btn_visualize_orbit.clicked.connect(self.open_orbit_visualization)
        self.btn_visualize_orbit.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.orbit_button_layout.addStretch()
        self.orbit_button_layout.addWidget(self.btn_visualize_orbit)
        self.orbit_button_layout.addStretch()
        self.data_viewer_layout.addLayout(self.orbit_button_layout)
        
        self.modelosLayout.addWidget(self.data_viewer_frame)
        
        # ========== PESTAÑA DE TRAIN (NEW) ==========
        self.tabTrain = QWidget()
        self.tabTrain.setObjectName("tabTrain")
        self.trainLayout = QVBoxLayout(self.tabTrain)
        self.labelTrain = QLabel("Training: configure training datasets and start training")
        self.labelTrain.setFont(QFont("Arial", 10, QFont.Bold))
        self.trainLayout.addWidget(self.labelTrain)
        
        # Training controls (Features and Labels)
        self.train_controls_frame = QFrame()
        self.train_controls_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.train_controls_frame.setLineWidth(2)
        self.train_controls_layout = QHBoxLayout(self.train_controls_frame)

        # Features and Labels selectors (split dataset)
        # Features selector
        self.label_features = QLabel("Features CSV:")
        self.train_controls_layout.addWidget(self.label_features)

        self.features_path_label = QLabel("<no file selected>")
        self.features_path_label.setMinimumWidth(260)
        self.train_controls_layout.addWidget(self.features_path_label)

        self.btn_select_features = QPushButton('Select Features')
        self.btn_select_features.clicked.connect(self.select_features_file)
        self.train_controls_layout.addWidget(self.btn_select_features)

        self.btn_load_features = QPushButton('Load Features')
        self.btn_load_features.clicked.connect(self.load_features_clicked)
        self.btn_load_features.setEnabled(False)
        self.train_controls_layout.addWidget(self.btn_load_features)

        # Labels selector
        self.label_labels = QLabel("Labels CSV:")
        self.train_controls_layout.addWidget(self.label_labels)

        self.labels_path_label = QLabel("<no file selected>")
        self.labels_path_label.setMinimumWidth(260)
        self.train_controls_layout.addWidget(self.labels_path_label)

        self.btn_select_labels = QPushButton('Select Labels')
        self.btn_select_labels.clicked.connect(self.select_labels_file)
        self.train_controls_layout.addWidget(self.btn_select_labels)

        self.btn_load_labels = QPushButton('Load Labels')
        self.btn_load_labels.clicked.connect(self.load_labels_clicked)
        self.btn_load_labels.setEnabled(False)
        self.train_controls_layout.addWidget(self.btn_load_labels)

        self.train_controls_layout.addStretch()

        # Outdir selector + start button (features/labels based training)
        self.btn_select_outdir = QPushButton('Select Output Dir')
        self.btn_select_outdir.clicked.connect(self.select_outdir)
        self.train_controls_layout.addWidget(self.btn_select_outdir)

        self.btn_start_training = QPushButton('Start Training')
        self.btn_start_training.clicked.connect(self.start_training_clicked)
        # disabled until features and labels are loaded
        self.btn_start_training.setEnabled(False)
        self.train_controls_layout.addWidget(self.btn_start_training)

        self.train_controls_layout.addStretch()
        self.trainLayout.addWidget(self.train_controls_frame)

        # Random Forest parameters
        self.rf_params_frame = QFrame()
        self.rf_params_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.rf_params_frame.setLineWidth(2)
        self.rf_params_layout = QHBoxLayout(self.rf_params_frame)

        # n_estimators
        self.label_n_estimators = QLabel("n_estimators:")
        self.label_n_estimators.setToolTip("Number of trees in the Random Forest")
        self.rf_params_layout.addWidget(self.label_n_estimators)
        self.spin_n_estimators = QSpinBox()
        self.spin_n_estimators.setRange(1, 10000)
        self.spin_n_estimators.setValue(100)
        self.spin_n_estimators.setToolTip("Number of trees (default 100)")
        self.spin_n_estimators.setMinimumWidth(80)
        self.rf_params_layout.addWidget(self.spin_n_estimators)

        # max_depth
        self.label_max_depth = QLabel("max_depth:")
        self.label_max_depth.setToolTip("Maximum depth of each tree (0 for None)")
        self.rf_params_layout.addWidget(self.label_max_depth)
        self.spin_max_depth = QSpinBox()
        self.spin_max_depth.setRange(-1, 1000)
        self.spin_max_depth.setValue(100)
        self.spin_max_depth.setToolTip("Maximum depth (0 means None)")
        self.spin_max_depth.setMinimumWidth(80)
        self.rf_params_layout.addWidget(self.spin_max_depth)

        # random_state
        self.label_random_state = QLabel("random_state:")
        self.label_random_state.setToolTip("Random seed for reproducibility (-1 for None)")
        self.rf_params_layout.addWidget(self.label_random_state)
        self.spin_random_state = QSpinBox()
        self.spin_random_state.setRange(-1, 999999)
        self.spin_random_state.setValue(42)
        self.spin_random_state.setToolTip("Random seed (-1 means None)")
        self.spin_random_state.setMinimumWidth(80)
        self.rf_params_layout.addWidget(self.spin_random_state)

        # n_jobs
        self.label_n_jobs = QLabel("n_jobs:")
        self.label_n_jobs.setToolTip("Number of parallel jobs (0 for None, -1 for all cores)")
        self.rf_params_layout.addWidget(self.label_n_jobs)
        self.spin_n_jobs = QSpinBox()
        self.spin_n_jobs.setRange(-1, 64)
        self.spin_n_jobs.setValue(-1)
        self.spin_n_jobs.setToolTip("Parallel jobs (-1 use all cores)")
        self.spin_n_jobs.setMinimumWidth(80)
        self.rf_params_layout.addWidget(self.spin_n_jobs)

        self.rf_params_layout.addStretch()
        self.trainLayout.addWidget(self.rf_params_frame)

        # Dataset preview frame
        self.dataset_preview_frame = QFrame()
        self.dataset_preview_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.dataset_preview_frame.setLineWidth(2)
        self.dataset_preview_layout = QVBoxLayout(self.dataset_preview_frame)

        # Header + pagination controls
        self.dataset_header_layout = QHBoxLayout()
        self.label_dataset_preview = QLabel("Dataset preview:")
        self.dataset_header_layout.addWidget(self.label_dataset_preview)
        self.dataset_header_layout.addStretch()

        self.dataset_pagination_widget = QWidget()
        self.dataset_pagination_layout = QHBoxLayout(self.dataset_pagination_widget)
        self.dataset_pagination_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_dataset_first = QPushButton('<<')
        self.btn_dataset_first.setMaximumWidth(40)
        self.btn_dataset_first.clicked.connect(self.go_to_first_dataset_page)
        self.btn_dataset_first.setEnabled(False)

        self.btn_dataset_prev = QPushButton('<')
        self.btn_dataset_prev.setMaximumWidth(40)
        self.btn_dataset_prev.clicked.connect(self.go_to_prev_dataset_page)
        self.btn_dataset_prev.setEnabled(False)

        self.label_dataset_page_info = QLabel('Page 0 of 0')
        self.label_dataset_page_info.setAlignment(Qt.AlignCenter)
        self.label_dataset_page_info.setMinimumWidth(100)

        self.btn_dataset_next = QPushButton('>')
        self.btn_dataset_next.setMaximumWidth(40)
        self.btn_dataset_next.clicked.connect(self.go_to_next_dataset_page)
        self.btn_dataset_next.setEnabled(False)

        self.btn_dataset_last = QPushButton('>>')
        self.btn_dataset_last.setMaximumWidth(40)
        self.btn_dataset_last.clicked.connect(self.go_to_last_dataset_page)
        self.btn_dataset_last.setEnabled(False)

        self.dataset_pagination_layout.addWidget(self.btn_dataset_first)
        self.dataset_pagination_layout.addWidget(self.btn_dataset_prev)
        self.dataset_pagination_layout.addWidget(self.label_dataset_page_info)
        self.dataset_pagination_layout.addWidget(self.btn_dataset_next)
        self.dataset_pagination_layout.addWidget(self.btn_dataset_last)

        self.dataset_header_layout.addWidget(self.dataset_pagination_widget)
        self.dataset_pagination_widget.setVisible(False)

        self.dataset_preview_layout.addLayout(self.dataset_header_layout)

        self.table_dataset_preview = QTableWidget()
        self.table_dataset_preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_dataset_preview.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_dataset_preview.setAlternatingRowColors(True)
        self.dataset_preview_layout.addWidget(self.table_dataset_preview)
        self.trainLayout.addWidget(self.dataset_preview_frame)

        # Labels preview
        self.labels_preview_frame = QFrame()
        self.labels_preview_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.labels_preview_frame.setLineWidth(2)
        self.labels_preview_layout = QVBoxLayout(self.labels_preview_frame)

        self.label_labels_preview_header = QLabel("Labels preview:")
        self.labels_preview_layout.addWidget(self.label_labels_preview_header)

        self.table_labels_preview = QTableWidget()
        self.table_labels_preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_labels_preview.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_labels_preview.setAlternatingRowColors(True)
        self.table_labels_preview.setMinimumHeight(120)
        self.labels_preview_layout.addWidget(self.table_labels_preview)

        self.trainLayout.addWidget(self.labels_preview_frame)

        # dataset paging state
        self._dataset_df = None
        self._dataset_current_page = 0
        self._dataset_rows_per_page = 500

        self.trainLayout.addStretch()
        # Add Train first
        self.tabWidget.addTab(self.tabTrain, "Train")

        # Rename 'Model' tab to 'Evaluation'
        self.tabWidget.addTab(self.tabModelos, "Model Prediction")
        
        self.tabWidget.addTab(self.tabTrain, "Model Training")
        
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
        self.table_database.setSortingEnabled(False)
        self.db_viewer_layout.addWidget(self.table_database)
        
        self.datosLayout.addWidget(self.db_viewer_frame)
        
        # Rename 'Data' tab to 'Visualize'
        self.tabWidget.addTab(self.tabDatos, "DataBase View")
        
        # Agregar el TabWidget al layout principal
        self.mainLayout.addWidget(self.tabWidget)
        
        self.setCentralWidget(self.centralwidget)
        
        # Barra de estado
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        
        QtCore.QMetaObject.connectSlotsByName(self)
        

        loading_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LoadingScreen.png')
        splash_pixmap = QPixmap(loading_img)
        splash = QSplashScreen(splash_pixmap, Qt.WindowStaysOnTopHint)
        splash.show()

        # Process events to ensure splash screen is displayed
        QApplication.processEvents()

        # Perform loading operations
        self.refresh_database()

        # Show main window and close splash screen after a short delay
        QtCore.QTimer.singleShot(1, lambda: self._show_and_finish_splash(splash))
        
    def _show_and_finish_splash(self, splash):
        splash.finish(self)
        self.show()
    
    def setup_window_properties(self):
        """Configura las propiedades de la ventana: posición, tamaño y redimensionamiento"""
        
        # Obtener las dimensiones de la pantalla
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        # Definir el tamaño de la ventana (80% del tamaño de la pantalla)
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        # Calcular la posición para centrar la ventana
        x_pos = (screen_width - window_width) // 2
        y_pos = (screen_height - window_height) // 2
        
        # Establecer geometría: (x, y, width, height)
        self.setGeometry(x_pos, y_pos, window_width, window_height)
        
        # Permitir redimensionamiento
        self.setMinimumSize(800, 600) 
        # self.setMaximumSize(1600, 1200) # Opcional
        self.setWindowTitle("Spes Nova Finder")
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo.png')
        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("Spes Nova Finder")

    # === Save-model dialog handler ===
    def on_request_model_save(self, suggested_path: str):
        """Show a Save File dialog asking where to persist the trained model.

        When the user selects a path, emit `model_save_path_chosen` with the
        absolute path so the backend can persist the model.
        """
        try:
            path, _ = QFileDialog.getSaveFileName(self, "Save trained model as", suggested_path or "model.joblib", "Joblib files (*.joblib);;All files (*.*)")
            if not path:
                # user cancelled - emit empty string to indicate cancel
                self.model_save_path_chosen.emit("")
                return
            # emit absolute path
            abs_path = os.path.abspath(path)
            self.model_save_path_chosen.emit(abs_path)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to open save dialog: {e}")

    # ========== ORBIT VISUALIZATION METHODS ==========
    
    def on_exoplanet_row_selected(self):
        """Enable orbit visualization button when a row is selected in the table."""
        selected_items = self.table_exoplanets.selectedItems()
        self.btn_visualize_orbit.setEnabled(len(selected_items) > 0 and self.visualization_dataframe is not None)
    
    def open_orbit_visualization(self):
        """Open a new window with the orbit visualization for the selected exoplanet."""
        if self.visualization_dataframe is None:
            QMessageBox.warning(self, "No Data", "No exoplanet data loaded.")
            return
        
        selected_rows = self.table_exoplanets.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select an exoplanet row to visualize.")
            return
        
        # Get the selected row index (in the current page view)
        view_row_index = selected_rows[0].row()
        
        # Calculate the actual DataFrame row index considering pagination
        actual_row_index = self.current_page * self.rows_per_page + view_row_index
        
        if actual_row_index >= len(self.visualization_dataframe):
            QMessageBox.warning(self, "Invalid Selection", "Selected row is out of range.")
            return
        
        try:
            # Import orbit module
            from backend import orbit
            
            # Create a new window for the orbit visualization
            orbit_window = QMainWindow(self)
            orbit_window.setWindowTitle(f"Orbit Visualization - Row {actual_row_index}")
            orbit_window.resize(1200, 800)
            
            # Create central widget and layout
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Create orbit visualization as a child widget
            # Pass parent=central_widget and run_app=False to embed it
            #sample = {'kepid':'KOI-0001','koi_period':54.4183827,'koi_time0bk':162.51384,
              #'koi_smass':1.0,'koi_srad':1.0,'koi_prad':1.00,'koi_sma':0.2734,
              #'koi_eccen':0.05,'koi_incl':89.57,'koi_longp':90.0,'koi_steff':5778.0}
            #df_sample = pd.DataFrame([sample])
            orbit_ctx = orbit.create_child_widget(
                df=self.visualization_dataframe,
                row_index=actual_row_index,
                speed=5.0,
                show_solar_system=True,
                show_habitable_zone=True,
                parent=central_widget
            )

            # Add the canvas to the layout
            layout.addWidget(orbit_ctx['canvas'].native)
            
            orbit_window.setCentralWidget(central_widget)
            
            # Store reference to prevent garbage collection
            self.orbit_windows.append(orbit_window)
            
            # Show the window
            orbit_window.show()
            
            self.statusbar.showMessage(f"Opened orbit visualization for row {actual_row_index}", 3000)
            
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Could not import orbit module. Make sure it's in the backend folder.\n\nError: {e}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Error creating orbit visualization:\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

    # ========== MÉTODOS ORIGINALES (PESTAÑA MODEL PREDICTION) ==========
    
    def open_csv_batch(self):
        """Open a file dialog to select a CSV batch and perform basic validation."""
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

    def select_features_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select features CSV", "", "CSV Files (*.csv);;All files (*)")
        if file_path:
            self._features_path = file_path
            self.features_path_label.setText(os.path.basename(file_path))
            ext = os.path.splitext(file_path)[1].lower()
            supported = ['.csv', '.tsv', '.tab', '.vot', '.votable', '.xml', '.tbl']
            self.btn_load_features.setEnabled(ext in supported)

    def select_labels_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select labels CSV", "", "CSV Files (*.csv);;All files (*)")
        if file_path:
            self._labels_path = file_path
            self.labels_path_label.setText(os.path.basename(file_path))
            ext = os.path.splitext(file_path)[1].lower()
            supported = ['.csv', '.tsv', '.tab', '.vot', '.votable', '.xml', '.tbl']
            self.btn_load_labels.setEnabled(ext in supported)

    def load_features_clicked(self):
        path = getattr(self, '_features_path', None)
        if not path:
            QMessageBox.warning(self, "No file", "No features file selected.")
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == '.csv':
                df, status, msg = dataio.loadcsvfile(path)
            elif ext in ['.tsv', '.tab']:
                df, status, msg = dataio.loadtabseptable(path)
            elif ext in ['.vot', '.votable', '.xml']:
                df, status, msg = dataio.loadvotableable(path)
            elif ext == '.tbl':
                df, status, msg = dataio.loadipactable(path)
            else:
                df, status, msg = dataio.loadcsvfile(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Error calling data loader: {e}")
            return

        if status != 1 or df is None:
            QMessageBox.critical(self, "Load failed", f"Failed to load features:\n{msg}")
            return

        self._features_df = df
        self.statusbar.showMessage(f"Features loaded: {len(df)} rows", 2000)
        self.display_dataset_preview(df)
        # enable start if labels already loaded
        if getattr(self, '_labels_df', None) is not None:
            self.btn_start_training.setEnabled(True)

    def load_labels_clicked(self):
        path = getattr(self, '_labels_path', None)
        if not path:
            QMessageBox.warning(self, "No file", "No labels file selected.")
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == '.csv':
                df, status, msg = dataio.loadcsvfile(path)
            elif ext in ['.tsv', '.tab']:
                df, status, msg = dataio.loadtabseptable(path)
            elif ext in ['.vot', '.votable', '.xml']:
                df, status, msg = dataio.loadvotableable(path)
            elif ext == '.tbl':
                df, status, msg = dataio.loadipactable(path)
            else:
                df, status, msg = dataio.loadcsvfile(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Error calling data loader: {e}")
            return

        if status != 1 or df is None:
            QMessageBox.critical(self, "Load failed", f"Failed to load labels:\n{msg}")
            return

        self._labels_df = df
        self.statusbar.showMessage(f"Labels loaded: {len(df)} rows", 2000)
        self.display_labels_preview(df)
        # enable start if features already loaded
        if getattr(self, '_features_df', None) is not None:
            self.btn_start_training.setEnabled(True)

    def display_labels_preview(self, dataframe: pd.DataFrame, max_rows: int = 200):
        if dataframe is None or dataframe.empty:
            self.table_labels_preview.setRowCount(0)
            self.table_labels_preview.setColumnCount(0)
            return

        df = dataframe if len(dataframe) <= max_rows else dataframe.head(max_rows)
        self.table_labels_preview.setRowCount(len(df))
        self.table_labels_preview.setColumnCount(len(df.columns))
        self.table_labels_preview.setHorizontalHeaderLabels(df.columns.tolist())

        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                self.table_labels_preview.setItem(i, j, item)

        self.table_labels_preview.resizeColumnsToContents()

    def select_outdir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output directory", "")
        if folder:
            self._out_dir = folder
            self.statusbar.showMessage(f"Output folder set", 2000)

    def start_training_clicked(self):
        # Training will use the loaded features and labels tables. We split
        # the features into confirmed/rejected CSVs based on labels and pass
        # their paths to the backend training API.
        X = getattr(self, '_features_df', None)
        y_df = getattr(self, '_labels_df', None)
        outdir = getattr(self, '_out_dir', None)

        if X is None or y_df is None:
            QMessageBox.warning(self, "Missing data", "Please load both features and labels before training.")
            return

        # resolve labels series
        if y_df.shape[1] == 1:
            y = y_df.iloc[:, 0].squeeze()
        elif 'koi_disposition' in y_df.columns:
            y = y_df['koi_disposition'].squeeze()
        else:
            y = y_df.iloc[:, 0].squeeze()

        if len(y) != len(X):
            QMessageBox.warning(self, "Mismatched sizes", "Features and labels must have the same number of rows.")
            return

        # Split into confirmed/rejected based on label values (1 / 0)
        try:
            mask_conf = (y == 1)
        except Exception:
            mask_conf = (y.astype(int) == 1)

        X_conf = X[mask_conf].copy()
        X_rej = X[~mask_conf].copy()

        # Determine output directory for temporary CSVs
        if outdir and os.path.isdir(outdir):
            tmp_dir = outdir
        else:
            import tempfile
            tmp_dir = tempfile.mkdtemp(prefix='hf_train_')

        confirmed_path = os.path.join(tmp_dir, 'confirmed_train.csv')
        rejected_path = os.path.join(tmp_dir, 'rejected_train.csv')
        X_conf.to_csv(confirmed_path, index=False)
        X_rej.to_csv(rejected_path, index=False)

        params = self.get_rf_params()
        self.start_training_signal.emit(confirmed_path, rejected_path, tmp_dir, params)
        self.statusbar.showMessage("Training started in background", 2000)

    def get_rf_params(self):
        """Return a dict with Random Forest parameters from the UI controls."""
        try:
            n_estimators = int(self.spin_n_estimators.value())
            max_depth = int(self.spin_max_depth.value())
            max_depth = None if max_depth == 0 else max_depth
            random_state = int(self.spin_random_state.value())
            random_state = None if random_state == -1 else random_state
            n_jobs = int(self.spin_n_jobs.value())
            n_jobs = None if n_jobs == 0 else n_jobs
            return {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'random_state': random_state,
                'n_jobs': n_jobs
            }
        except Exception:
            return {'n_estimators': 100, 'max_depth': None, 'random_state': None, 'n_jobs': 1}

    def select_dataset_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select dataset CSV", "", "CSV Files (*.csv);;All files (*)")
        if not file_path:
            return
        self._dataset_path = file_path
        self.dataset_path_label.setText(os.path.basename(file_path))
        ext = os.path.splitext(file_path)[1].lower()
        supported = ['.csv', '.tsv', '.tab', '.vot', '.votable', '.xml', '.tbl']
        if ext in supported:
            self.btn_load_dataset.setEnabled(True)
        else:
            self.btn_load_dataset.setEnabled(False)
            QMessageBox.warning(self, "Invalid file", "Please select a supported table file (CSV/TSV/VOTable/IPAC).")

    def load_dataset_clicked(self):
        """Load the selected dataset using analysis.dataio and preview it."""
        path = getattr(self, '_dataset_path', None)
        if not path:
            QMessageBox.warning(self, "No file", "No dataset selected. Use 'Select Dataset' first.")
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == '.csv':
                df, status, msg = dataio.loadcsvfile(path)
            elif ext in ['.tsv', '.tab']:
                df, status, msg = dataio.loadtabseptable(path)
            elif ext in ['.vot', '.votable', '.xml']:
                df, status, msg = dataio.loadvotableable(path)
            elif ext == '.tbl':
                df, status, msg = dataio.loadipactable(path)
            else:
                df, status, msg = dataio.loadcsvfile(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Error calling data loader: {e}")
            return

        if status != 1 or df is None:
            QMessageBox.critical(self, "Load failed", f"Failed to load dataset:\n{msg}")
            return

        self._dataset_df = df
        self.statusbar.showMessage(f"Dataset loaded: {len(df)} rows", 2000)
        self.display_dataset_preview(df)

    def display_dataset_preview(self, dataframe: pd.DataFrame, max_rows: int = 200):
        """Store the dataset and render page 0 into the preview table."""
        if dataframe is None or dataframe.empty:
            self._dataset_df = None
            self.table_dataset_preview.setRowCount(0)
            self.table_dataset_preview.setColumnCount(0)
            self.dataset_pagination_widget.setVisible(False)
            return

        self._dataset_df = dataframe.copy()
        self._dataset_current_page = 0

        total_rows = len(self._dataset_df)
        total_pages = (total_rows + self._dataset_rows_per_page - 1) // self._dataset_rows_per_page
        if total_pages > 1:
            self.dataset_pagination_widget.setVisible(True)
        else:
            self.dataset_pagination_widget.setVisible(False)

        self.load_current_dataset_page()

    def load_current_dataset_page(self):
        """Render current dataset page into the QTableWidget and update nav state."""
        if self._dataset_df is None or self._dataset_df.empty:
            return

        total_rows = len(self._dataset_df)
        total_pages = (total_rows + self._dataset_rows_per_page - 1) // self._dataset_rows_per_page

        start_row = self._dataset_current_page * self._dataset_rows_per_page
        end_row = min(start_row + self._dataset_rows_per_page, total_rows)

        page_data = self._dataset_df.iloc[start_row:end_row]

        self.label_dataset_preview.setText(f"Dataset preview (Showing {start_row+1}-{end_row} of {total_rows})")
        self.label_dataset_page_info.setText(f'Page {self._dataset_current_page + 1} of {total_pages}')

        self.table_dataset_preview.setRowCount(len(page_data))
        self.table_dataset_preview.setColumnCount(len(page_data.columns))
        self.table_dataset_preview.setHorizontalHeaderLabels(page_data.columns.tolist())

        for i in range(len(page_data)):
            for j, col in enumerate(page_data.columns):
                value = page_data.iloc[i, j]
                item = QTableWidgetItem(str(value))
                self.table_dataset_preview.setItem(i, j, item)

        self.table_dataset_preview.resizeColumnsToContents()

        self.btn_dataset_first.setEnabled(self._dataset_current_page > 0)
        self.btn_dataset_prev.setEnabled(self._dataset_current_page > 0)
        self.btn_dataset_next.setEnabled(self._dataset_current_page < total_pages - 1)
        self.btn_dataset_last.setEnabled(self._dataset_current_page < total_pages - 1)

        self.dataset_pagination_widget.setVisible(total_pages > 1)

    def go_to_first_dataset_page(self):
        self._dataset_current_page = 0
        self.load_current_dataset_page()

    def go_to_prev_dataset_page(self):
        if self._dataset_current_page > 0:
            self._dataset_current_page -= 1
            self.load_current_dataset_page()

    def go_to_next_dataset_page(self):
        if self._dataset_df is not None:
            total_pages = (len(self._dataset_df) + self._dataset_rows_per_page - 1) // self._dataset_rows_per_page
            if self._dataset_current_page < total_pages - 1:
                self._dataset_current_page += 1
                self.load_current_dataset_page()

    def go_to_last_dataset_page(self):
        if self._dataset_df is not None:
            total_pages = (len(self._dataset_df) + self._dataset_rows_per_page - 1) // self._dataset_rows_per_page
            self._dataset_current_page = total_pages - 1
            self.load_current_dataset_page()
    
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
        """Remove the selected batch entry from the list and notify backend."""
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
        """Clear all loaded batches after user confirmation."""
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
        self.btn_visualize_orbit.setEnabled(False)
    
    def save_to_database(self):
        items_seleccionados = self.lista_archivos.selectedItems()
        if not items_seleccionados:
            return
        
        item = items_seleccionados[0]
        self.save_to_database_signal.emit(item.data(Qt.UserRole))
        
    def start_prediction(self):
        """Start prediction for the selected batch or for all loaded batches."""
        items_seleccionados = self.lista_archivos.selectedItems()
        if items_seleccionados:
            item = items_seleccionados[0]
            batch_id = item.data(Qt.UserRole)
            self.start_prediction_signal.emit(batch_id)
            self.statusbar.showMessage(f"Prediction started for Batch {batch_id}", 2000)
            return

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
            for i in range(self.lista_archivos.count()):
                item = self.lista_archivos.item(i)
                batch_id = item.data(Qt.UserRole)
                self.start_prediction_signal.emit(batch_id)
            self.statusbar.showMessage("Prediction started for all loaded batches", 2000)

    def load_model_clicked(self):
        """Open a file dialog to choose a model (joblib)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model file",
            "",
            "Joblib files (*.joblib);;All files (*.*)"
        )

        if not file_path:
            return

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
                self.load_model_signal.emit(file_path, -1)
        else:
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
        """Enable or disable UI buttons depending on the current selection state."""
        if self.lista_archivos.selectedItems():
            self.btn_start_prediction.setEnabled(True)
            self.btn_save_to_db.setEnabled(True)
            self.btn_load_model.setEnabled(True)
        else:
            self.btn_start_prediction.setEnabled(False)
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
        """Handle backend prediction progress updates."""
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
            self.btn_start_prediction.setEnabled(False)
            self.btn_save_to_db.setEnabled(False)
        else:
            if status == 'completed':
                self.btn_start_prediction.setEnabled(True)
                self.btn_save_to_db.setEnabled(True)
    
    def display_batch_data(self, dataframe: pd.DataFrame, vis_dataframe: pd.DataFrame, batch_id: int):
        """Display a batch DataFrame in the table widget with pagination."""
        if dataframe is None or dataframe.empty:
            self.clear_table_view()
            self.label_data_viewer.setText(f"Batch {batch_id}: No data available")
            return
        
        # Guardar referencia al dataframe completo
        self.current_dataframe = dataframe
        self.visualization_dataframe = vis_dataframe
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
        """Render database DataFrame into the database table view."""
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
        """Sort the currently displayed database page using a background thread."""
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
        """Perform the search/filter operation in a background thread."""
        search_text = self.search_input.text().strip()
        
        if self.database_dataframe is None:
            return
        
        # Si ya hay un thread corriendo, esperar a que termine
        if self.search_thread and self.search_thread.isRunning():
            # En lugar de retornar, programar otra búsqueda
            self.search_timer.start(200)
            return
        
        if search_text != "" and self.combo_sort_column.currentText() == "":
            return
        
        # Mostrar indicador de carga
        self.label_loading.setVisible(True)
        
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
    
    def clear_search(self):
        """Clear the search input and trigger a refresh via the debounce timer."""
        self.search_input.clear()
    
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

    # === Graphs window ===
    def open_graphs_window(self, image_paths: list):
        """Open a simple window that displays a list of image paths.

        The window will show thumbnails and allow clicking to open full-size.
        """
        if not image_paths:
            QMessageBox.information(self, "No graphs", "No graphs were produced.")
            return

        gw = QWidget()
        gw.setWindowTitle('Training visualizations')
        layout = QVBoxLayout(gw)
        scroll = QScrollArea()
        container = QWidget()
        v = QVBoxLayout(container)

        for p in image_paths:
            if not os.path.exists(p):
                continue
            lbl = QLabel(os.path.basename(p))
            pix = QPixmap(p)
            if pix.isNull():
                continue
            # scale down thumbnail
            thumb = pix.scaledToWidth(600)
            img_label = QLabel()
            img_label.setPixmap(thumb)
            img_label.setToolTip(p)
            v.addWidget(lbl)
            v.addWidget(img_label)

        container.setLayout(v)
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        gw.resize(800, 600)
        gw.show()
        # keep reference to avoid garbage collection
        self._graphs_window = gw


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = MainWindow()
    sys.exit(app.exec_())
