from PyQt5.QtWidgets import QApplication
from gui.gui import MainWindow

def main():
    import sys
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()