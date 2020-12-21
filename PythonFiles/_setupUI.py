import sys
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QGroupBox, QTextEdit, QMessageBox, QSizePolicy, \
    QFileDialog
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import ctypes
from matplotlib.figure import Figure
import tensorflow as tf


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.noise_dim = 100
        self.num_exp_to_generate = 4
        self.model = tf.keras.models.load_model('../save/WGAN_cartoon_64_0021.h5')
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()

    def plot(self):
        # self.axes.axis('off')
        self.figure.clear()
        seed = tf.random.normal([self.num_exp_to_generate, self.noise_dim])
        predictions = self.model(seed, training=False)
        try:
            for i in range(predictions.shape[0]):
                ax = self.figure.add_subplot(2, 2, i + 1)
                ax.imshow((predictions[i, :, :, :] + 1) / 2)
                ax.axis('off')
                self.draw()
            self.figure.subplots_adjust(wspace=0, hspace=0)

            window.textEdit.setPlainText(str(seed))

        except Exception as e:
            pass

    def clear_image(self):
        try:
            self.figure.clear()
            self.draw()
        except Exception as e:
            print(str(e))

    def save_image(self):
        try:
            fileName = QFileDialog.getSaveFileName(self, '保存文件', '.', '图像文件(*.png *.jpg)')
            path = fileName[0]
            print(str(path))
            self.figure.savefig(str(path))
        except Exception as e:
            print(str(e))


class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.setupUI()
        self.show()

    def setupUI(self):
        self.resize(1290, 861)
        self.setWindowTitle('Create AnimeFaces With DCGan!')
        self.setWindowIcon(QIcon('../ico/favicon.ico'))

        m = PlotCanvas(self)
        m.move(130, 110)
        # self.image_shower.setGeometry(20, 10, 600, 600)

        self.draw_image_btn = QPushButton('生成图像', self)
        self.draw_image_btn.setGeometry(260, 560, 131, 41)
        self.draw_image_btn.clicked.connect(m.plot)
        self.draw_image_btn.setStyleSheet("""
            .QPushButton {
        box-shadow: 0px 1px 0px 0px #f0f7fa;
        background:linear-gradient(to bottom, #33bdef 5%, #019ad2 100%);
        background-color:#33bdef;
        border-radius:6px;
        border:1px solid #057fd0;
        display:inline-block;
        cursor:pointer;
        color:#ffffff;
        font-family:Arial;
        font-size:15px;
        font-weight:bold;
        padding:6px 24px;
        text-decoration:none;
        text-shadow:0px -1px 0px #5b6178;
    }
    .QPushButton:hover {
        background:linear-gradient(to bottom, #019ad2 5%, #33bdef 100%);
        background-color:#019ad2;
    }
    .myButton:active {
        position:relative;
        top:1px;
    }
        """)

        self.clear_image_btn = QPushButton('清除图像', self)
        self.clear_image_btn.setGeometry(40, 720, 137, 39)
        self.clear_image_btn.setStyleSheet("""
            .QPushButton {
        box-shadow:inset 0px 1px 0px 0px #f7c5c0;
        background:linear-gradient(to bottom, #fc8d83 5%, #e4685d 100%);
        background-color:#fc8d83;
        border-radius:6px;
        border:1px solid #d83526;
        display:inline-block;
        cursor:pointer;
        color:#ffffff;
        font-family:Arial;
        font-size:15px;
        font-weight:bold;
        padding:6px 24px;
        text-decoration:none;
        text-shadow:0px 1px 0px #b23e35;
    }
    .QPushButton:hover {
        background:linear-gradient(to bottom, #e4685d 5%, #fc8d83 100%);
        background-color:#e4685d;
    }
    .QPushButton:active {
        position:relative;
        top:1px;
    }
        """)
        self.clear_image_btn.clicked.connect(m.clear_image)

        self.extract_feature_btn = QPushButton('提取特征', self)
        self.extract_feature_btn.setGeometry(185, 720, 137, 39)
        self.extract_feature_btn.setStyleSheet("""
                .QPushButton {
            box-shadow:inset 0px 1px 0px 0px #bbdaf7;
            background:linear-gradient(to bottom, #79bbff 5%, #378de5 100%);
            background-color:#79bbff;
            border-radius:6px;
            border:1px solid #84bbf3;
            display:inline-block;
            cursor:pointer;
            color:#ffffff;
            font-family:Arial;
            font-size:15px;
            font-weight:bold;
            padding:6px 24px;
            text-decoration:none;
            text-shadow:0px 1px 0px #528ecc;
        }
        .QPushButton:hover {
            background:linear-gradient(to bottom, #378de5 5%, #79bbff 100%);
            background-color:#378de5;
        }
        .QPushButton:active {
            position:relative;
            top:1px;
        }
            """)
        self.extract_feature_btn.clicked.connect(self.extract_feature)

        self.save_image_btn = QPushButton('保存图像', self)
        self.save_image_btn.setGeometry(329, 720, 137, 39)
        self.save_image_btn.setStyleSheet("""
                .QPushButton {
            box-shadow:inset 0px 1px 0px 0px #d9fbbe;
            background:linear-gradient(to bottom, #b8e356 5%, #a5cc52 100%);
            background-color:#b8e356;
            border-radius:6px;
            border:1px solid #83c41a;
            display:inline-block;
            cursor:pointer;
            color:#ffffff;
            font-family:Arial;
            font-size:15px;
            font-weight:bold;
            padding:6px 24px;
            text-decoration:none;
            text-shadow:0px 1px 0px #86ae47;
        }
        .QPushButton:hover {
            background:linear-gradient(to bottom, #a5cc52 5%, #b8e356 100%);
            background-color:#a5cc52;
        }
        .QPushButton:active {
            position:relative;
            top:1px;
        }
            """)
        self.save_image_btn.clicked.connect(m.save_image)

        self.exit_app_btn = QPushButton('退出程序', self)
        self.exit_app_btn.setGeometry(473, 720, 137, 39)
        self.exit_app_btn.clicked.connect(self.close)
        self.exit_app_btn.setStyleSheet("""
                .QPushButton {
            box-shadow:inset 0px 1px 0px 0px #f5978e;
            background:linear-gradient(to bottom, #f24537 5%, #c62d1f 100%);
            background-color:#f24537;
            border-radius:6px;
            border:1px solid #d02718;
            display:inline-block;
            cursor:pointer;
            color:#ffffff;
            font-family:Arial;
            font-size:15px;
            font-weight:bold;
            padding:6px 24px;
            text-decoration:none;
            text-shadow:0px 1px 0px #810e05;
        }
        .QPushButton:hover {
            background:linear-gradient(to bottom, #c62d1f 5%, #f24537 100%);
            background-color:#c62d1f;
        }
        .QPushButton:active {
            position:relative;
            top:1px;
        }
            """)

        self.groupBox = QGroupBox('图像处理信息', self)
        self.groupBox.setStyleSheet(' border:1px solid #bfd1eb;background:#f3faff')
        self.groupBox.setGeometry(670, 20, 571, 711)

        self.textEdit = QTextEdit(self)
        self.textEdit.setEnabled(True)
        self.textEdit.setGeometry(700, 50, 521, 671)
        self.textEdit.setPlaceholderText("""        /*
                         ,----------------,              ,---------,
                    ,-----------------------,          ,"        ,"|
                  ,"                      ,"|        ,"        ,"  |
                 +-----------------------+  |      ,"        ,"    |
                 |  .-----------------.  |  |     +---------+      |
                 |  |                 |  |  |     | -==----'|      |
                 |  |  I LOVE DOS!    |  |  |     |         |      |
                 |  |  Bad command or |  |  |/----|`---=    |      |
                 |  |  C:\>_          |  |  |   ,/|==== ooo |      ;
                 |  |                 |  |  |  // |(((( [33]|    ,"
                 |  `-----------------'  |," .;'| |((((     |  ,"
                 +-----------------------+  ;;  | |         |,"
                    /_)______________(_/  //'   | +---------+
               ___________________________/___  `,
              /  oooooooooooooooo  .o.  oooo /,   \,"-----------
             / ==ooooooooooooooo==.o.  ooo= //   ,`\--{)B     ,"
            /_==__==========__==_ooo__ooo=_/'   /___________,"
             
            */""")
        self.textEdit.setStyleSheet("  border:1px solid #a5b6c8;background:#eef3f7")

        # self.clear_console_button = QPushButton('清除信息', self)
        # self.clear_console_button.setGeometry(920, 750, 121, 41)
        # self.clear_console_button.setStyleSheet("""
        #             .QPushButton {
        #     box-shadow:inset 0px 1px 0px 0px #efdcfb;
        #     background:linear-gradient(to bottom, #dfbdfa 5%, #bc80ea 100%);
        #     background-color:#dfbdfa;
        #     border-radius:6px;
        #     border:1px solid #c584f3;
        #     display:inline-block;
        #     cursor:pointer;
        #     color:#ffffff;
        #     font-family:Arial;
        #     font-size:15px;
        #     font-weight:bold;
        #     padding:6px 24px;
        #     text-decoration:none;
        #     text-shadow:0px 1px 0px #9752cc;
        # }
        # .QPushButton:hover {
        #     background:linear-gradient(to bottom, #bc80ea 5%, #dfbdfa 100%);
        #     background-color:#bc80ea;
        # }
        # .QPushButton:active {
        #     position:relative;
        #     top:1px;
        # }
        #     """)
        # self.clear_image_btn.clicked.connect(m.clear_console)

    # def clear_console(self):
    #     try:
    #         self.textEdit.clear()
    #     except Exception as e:
    #         print(str(e))
    def extract_feature(self):
        try:
            fileName = QFileDialog.getSaveFileName(self, '保存文件', '.', '文本文件(*.txt)')
            path = fileName[0]
            file = self.textEdit.toPlainText()
            f = open(path, 'w')
            f.write(file)
            f.close()
        except Exception as e:
            print(str(e))

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '退出程序', "真的要退出程序吗QAQ?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID()
    window = Window()
    sys.exit(app.exec_())
