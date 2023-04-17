import sys
from PyQt5.QtWidgets import QApplication, QWidget


if __name__ == '__main__':
    from rtsp_dialog import Ui_Form
else:
    from .rtsp_dialog import Ui_Form

class Window(QWidget, Ui_Form):
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
