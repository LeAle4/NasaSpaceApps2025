import sys
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import *
import pandas as pd
import os



class MainWindow(QMainWindow):
    load_csv_signal = pyqtSignal(str)
    remove_batch_signal = pyqtSignal(int)
    clear_batches_signal = pyqtSignal()
    start_prediction_signal = pyqtSignal(int)
    save_to_database_signal = pyqtSignal(int)
    def __init__(self):
        super().__init__()
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
        self.batch_files_frame.setMinimumHeight(250)
        
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
        self.btn_save_to_db.setEnabled(False)  # Placeholder for future functionality
        
        self.buttons_layout.addWidget(self.btn_remove)
        self.buttons_layout.addWidget(self.btn_clear)
        self.buttons_layout.addWidget(self.btn_save_to_db)
        self.buttons_layout.addStretch()
        
        self.frame_layout.addLayout(self.buttons_layout)
        
        self.modelosLayout.addWidget(self.batch_files_frame)
        
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
        self.modelosLayout.addStretch()
        
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
            
            if not file_path:  # Usuario canceló
                return None
                
            # Validar que es archivo CSV
            if not file_path.lower().endswith('.csv'):
                QMessageBox.warning(
                    self, 
                    "Formato incorrecto", 
                    "Por favor selecciona un archivo CSV"
                )
                return None
                
            # Validar que el archivo existe
            import os
            if not os.path.exists(file_path):
                QMessageBox.critical(
                    self,
                    "Archivo no encontrado",
                    f"El archivo no existe:\n{file_path}"
                )
                return None
                
            # Llamar al método de carga
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
    
    def save_to_database(self):
        items_seleccionados = self.lista_archivos.selectedItems()
        if not items_seleccionados:
            return
        
        item = items_seleccionados[0]
        self.save_to_database_signal.emit(item.data(Qt.UserRole))
        
    def start_prediction(self):
        pass

    def show_msg(self, tipo, mensaje):
        if tipo == "succes":
            QMessageBox.information(self, "Succes!", mensaje)
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
   


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec_())