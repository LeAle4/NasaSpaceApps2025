import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from vispy import app, scene
from vispy.geometry.generation import create_sphere

# Physical constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
M_SUN = 1.9891e30  # Mass of the sun in kg
R_SUN = 6.9634e8  # Radius of the sun in meters
R_EARTH = 6.371e6  # Radius of the Earth in meters
AU = 1.495978707e11 # Astronomical Unit in meters
DAY = 86400  # Seconds in a day
TWOPI = 2.0 * np.pi

def kepler_E_from_M(M, e, tol=1e-9, maxiter=80):
    M = np.mod(M, 2 * np.pi)
    if e < 0.8:
        E = M
        

app.use_app('pyqt5')

class VispyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Visualización 3D con VisPy')
        self.setGeometry(100, 100, 800, 600)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Crear el canvas de VisPy
        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600))
        layout.addWidget(self.canvas.native)
        
        # Configurar la vista 3D
        self.view = self.canvas.central_widget.add_view()
        
        # Crear una esfera de ejemplo
        mesh = create_sphere(20, 20, radius=1.0)
        vertices = mesh.get_vertices()
        faces = mesh.get_faces()
        
        # Agregar la esfera a la escena
        sphere_visual = scene.visuals.Mesh(vertices, faces, color='lightblue', parent=self.view.scene)
        
        # Agregar ejes de coordenadas
        scene.visuals.XYZAxis(parent=self.view.scene)
        
        # Configurar la cámara
        self.view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=30, elevation=30, distance=4)
        self.view.camera.set_range(x=(-2, 2), y=(-2, 2), z=(-2, 2))

def main():
    # Crear la aplicación PyQt
    qt_app = QApplication(sys.argv)
    
    # Crear y mostrar la ventana
    window = VispyWindow()
    window.show()
    
    # Ejecutar la aplicación
    sys.exit(qt_app.exec_())

if __name__ == '__main__':
    main()