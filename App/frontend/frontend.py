from PyQt5 import QtCore, QtGui, QtWidgets


class MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 480)
        
        # Widget central
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Layout principal
        self.mainLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        
        # QTabWidget para las pestañas
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        
        # Pestaña de Modelos
        self.tabModelos = QtWidgets.QWidget()
        self.tabModelos.setObjectName("tabModelos")
        self.modelosLayout = QtWidgets.QVBoxLayout(self.tabModelos)
        
        # Aquí puedes agregar widgets específicos para la pestaña de modelos
        self.labelModelos = QtWidgets.QLabel("Contenido de Modelos")
        self.labelModelos.setAlignment(QtCore.Qt.AlignCenter)
        self.modelosLayout.addWidget(self.labelModelos)
        
        self.tabWidget.addTab(self.tabModelos, "")
        
        # Pestaña de Datos
        self.tabDatos = QtWidgets.QWidget()
        self.tabDatos.setObjectName("tabDatos")
        self.datosLayout = QtWidgets.QVBoxLayout(self.tabDatos)
        
        # Aquí puedes agregar widgets específicos para la pestaña de datos
        self.labelDatos = QtWidgets.QLabel("Contenido de Datos")
        self.labelDatos.setAlignment(QtCore.Qt.AlignCenter)
        self.datosLayout.addWidget(self.labelDatos)
        
        self.tabWidget.addTab(self.tabDatos, "")
        
        # Agregar el TabWidget al layout principal
        self.mainLayout.addWidget(self.tabWidget)
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        # Barra de estado
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "HopeFinder"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabModelos), 
                                   _translate("MainWindow", "Modelos"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabDatos), 
                                   _translate("MainWindow", "Datos"))
        self.labelModelos.setText(_translate("MainWindow", "Contenido de Modelos"))
        self.labelDatos.setText(_translate("MainWindow", "Contenido de Datos"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = MainWindow()
    ui.setupUi(window)
    window.show()
    sys.exit(app.exec_())