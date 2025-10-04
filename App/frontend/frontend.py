import sys
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
import pandas as pd



class MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("HopeFinder")
        MainWindow.resize(640, 480)
        
        # Widget central
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Layout principal
        self.mainLayout = QVBoxLayout(self.centralwidget)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        
        # QTabWidget para las pestañas
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        
        # Pestaña de Modelos
        self.tabModelos = QWidget()
        self.tabModelos.setObjectName("tabModelos")
        self.modelosLayout = QVBoxLayout(self.tabModelos)
        
        # Aquí puedes agregar widgets específicos para la pestaña de modelos
        self.labelModelos = QLabel("Import exoplanet data:")
        self.labelModelos.setFont(QFont("Arial", 12)) 
        self.labelModelos.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.btn_abrir_csv = QPushButton('Open data file')
        self.btn_abrir_csv.clicked.connect(self.abrir_csv)
        self.modelosLayout.addWidget(self.labelModelos)
        self.modelosLayout.addWidget(self.btn_abrir_csv)
        self.modelosLayout.addStretch()
        
        self.tabWidget.addTab(self.tabModelos, "Model")
        
        # Pestaña de Datos
        self.tabDatos = QWidget()
        self.tabDatos.setObjectName("tabDatos")
        self.datosLayout = QVBoxLayout(self.tabDatos)
        
        # Aquí puedes agregar widgets específicos para la pestaña de datos
        self.labelDatos = QLabel("Contenido de Datos")
        self.labelDatos.setAlignment(Qt.AlignCenter)
        self.datosLayout.addWidget(self.labelDatos)
        
        self.tabWidget.addTab(self.tabDatos, "Data")
        
        # Agregar el TabWidget al layout principal
        self.mainLayout.addWidget(self.tabWidget)
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        # Barra de estado
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def abrir_csv(self):
        pass
   


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = MainWindow()
    ui.setupUi(window)
    window.show()
    sys.exit(app.exec_())