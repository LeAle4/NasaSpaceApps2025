import sys
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import *
import pandas as pd
import os


class DataLoaderThread(QThread):
    """Thread para cargar datos sin congelar la interfaz"""
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


class MainWindow(QMainWindow):
    load_csv_signal = pyqtSignal(str)
    remove_batch_signal = pyqtSignal(int)
    clear_batches_signal = pyqtSignal()
    start_prediction_signal = pyqtSignal(int)
    save_to_database_signal = pyqtSignal(int)
    request_batch_data_signal = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.current_dataframe = None
        self.current_batch_id = None
        self.current_page = 0
        self.rows_per_page = 500
        self.loader_thread = None
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
        
        # Pestaña de Modelos
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
        
        self.btn_remove = QPushButton('Remove Selected')
        self.btn_remove.clicked.connect(self.remover_archivo)
        self.btn_remove.setEnabled(False)
        
        self.btn_clear = QPushButton('Clear All')
        self.btn_clear.clicked.connect(self.limpiar_archivos)
        self.btn_clear.setEnabled(False)
        
        self.btn_save_to_db = QPushButton('Save to Database')
        self.btn_save_to_db.clicked.connect(self.save_to_database)
        self.btn_save_to_db.setEnabled(False)
        
        self.buttons_layout.addWidget(self.btn_remove)
        self.buttons_layout.addWidget(self.btn_clear)
        self.buttons_layout.addWidget(self.btn_save_to_db)
        self.buttons_layout.addStretch()
        
        self.frame_layout.addLayout(self.buttons_layout)
        
        self.modelosLayout.addWidget(self.batch_files_frame)
        
        # NUEVO: Frame para mostrar datos del batch seleccionado
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
        
        # Botones principales
        self.hbox = QHBoxLayout()
        
        # start prediction button
        self.btn_start_prediction = QPushButton('Start Prediction')
        self.btn_start_prediction.clicked.connect(self.start_prediction)
        self.btn_start_prediction.setEnabled(False)
        
        # open file button
        self.btn_abrir_csv = QPushButton('Add Data Batch')
        self.btn_abrir_csv.clicked.connect(self.open_csv_batch)
        
        self.hbox.addWidget(self.btn_start_prediction)
        self.hbox.addWidget(self.btn_abrir_csv)
        self.hbox.addStretch()
        
        self.modelosLayout.addLayout(self.hbox)
        
        self.tabWidget.addTab(self.tabModelos, "Model")
        
        # Pestaña de Datos
        self.tabDatos = QWidget()
        self.tabDatos.setObjectName("tabDatos")
        self.datosLayout = QVBoxLayout(self.tabDatos)
        
        self.labelDatos = QLabel("Contenido de Datos")
        self.labelDatos.setAlignment(Qt.AlignCenter)
        self.datosLayout.addWidget(self.labelDatos)
        
        self.tabWidget.addTab(self.tabDatos, "Data")
        
        # Agregar el TabWidget al layout principal
        self.mainLayout.addWidget(self.tabWidget)
        
        self.setCentralWidget(self.centralwidget)
        
        # Barra de estado
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        
        QtCore.QMetaObject.connectSlotsByName(self)

    def open_csv_batch(self):
        """Abre diálogo para seleccionar CSV con validaciones"""
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
        """Elimina el archivo seleccionado de la lista"""
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
        """Limpia todos los archivos de la lista"""
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
        """Limpia la vista de tabla y resetea controles"""
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
        pass

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
        """Habilita el botón de predicción si hay un archivo seleccionado"""
        if self.lista_archivos.selectedItems():
            self.btn_start_prediction.setEnabled(True)
            self.btn_save_to_db.setEnabled(True)
        else:
            self.btn_start_prediction.setEnabled(False)
        if self.lista_archivos.count() == 0:
            self.btn_clear.setEnabled(False)
            self.btn_remove.setEnabled(False)
    
    def on_batch_selected(self):
        """Se ejecuta cuando se selecciona un batch"""
        items_seleccionados = self.lista_archivos.selectedItems()
        if items_seleccionados:
            batch_id = items_seleccionados[0].data(Qt.UserRole)
            self.request_batch_data_signal.emit(batch_id)
    
    def display_batch_data(self, dataframe: pd.DataFrame, batch_id: int):
        """Muestra los datos del batch en la tabla con paginación"""
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
        """Carga la página actual de datos"""
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


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec_())