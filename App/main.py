from PyQt5.QtWidgets import QApplication
import sys
from frontend.frontend import MainWindow
from backend.backend import CurrentSesion
from PyQt5.QtWidgets import QFileDialog, QMessageBox


class HopeFinderApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.frontend = MainWindow()
        self.backend = CurrentSesion()
        self.frontend.setupUi()
    
    def connect_signals(self):
        self.frontend.load_csv_signal.connect(self.backend.newPredictionBatch)
        self.backend.popup_msg_signal.connect(self.frontend.show_msg)
        self.backend.batch_info_signal.connect(self.frontend.add_batch_info)
        self.frontend.remove_batch_signal.connect(self.backend.removeBatch)
        self.frontend.clear_batches_signal.connect(self.backend.clearBatches)
    
    def run(self):
        self.frontend.show()
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    app = HopeFinderApp()
    app.connect_signals()
    app.run()
    