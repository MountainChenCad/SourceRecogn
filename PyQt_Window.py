"""
该示例具有按钮和标签和文本浏览器。 通过按钮显示输入对话框以便获取值。 输入的文本将显示在窗口的标签和文本浏览器中。
"""
import datetime
from PyQt5.QtChart import *
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtWidgets, QtGui, QtCore

##########################################
# ui界面设置
from PyQt5.QtWidgets import QComboBox, QRadioButton, QButtonGroup, QLabel, QLineEdit, QTextEdit
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QInputDialog, QTextBrowser)
import sys
from fine_tune import main


class My_Window(QWidget):

    def __init__(self):
        super(My_Window, self).__init__()
        self.resize(1080, 900)

        self.initUI()

    def initUI(self):

        # 设置主体窗口
        self.setGeometry(420, 90, 1080, 900)  # 横坐标，纵坐标，长，宽
        self.setWindowTitle('跨场景信号调制方式识别系统')  # GUI界面名称

        # 设置文件选择及存放
        self.file = QtWidgets.QPushButton('选择源域文件路径', self)
        self.file1 = QtWidgets.QPushButton('选择目标域文件路径', self)
        self.file2 = QtWidgets.QPushButton('选择权重文件存储路径', self)
        self.file3 = QtWidgets.QPushButton('选择训练日志存储路径', self)
        self.file.setGeometry(QtCore.QRect(50, 50, 175, 28))  # （横坐标，纵坐标，长，宽），原点为左上角
        self.file1.setGeometry(QtCore.QRect(50, 150, 175, 28))
        self.file2.setGeometry(QtCore.QRect(50, 250, 175, 28))
        self.file3.setGeometry(QtCore.QRect(50, 350, 175, 28))
        self.file.setObjectName("file")
        self.file1.setObjectName("file1")
        self.file2.setObjectName("file2")
        self.file3.setObjectName("file3")
        self.file.setStyleSheet(
            "QPushButton{background-color:rgb(111,180,219)}"  # 按键背景色
            "QPushButton:hover{color:green}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )
        self.file1.setStyleSheet(
            "QPushButton{background-color:rgb(111,180,219)}"  # 按键背景色
            "QPushButton:hover{color:green}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )
        self.file2.setStyleSheet(
            "QPushButton{background-color:rgb(111,180,219)}"  # 按键背景色
            "QPushButton:hover{color:green}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )
        self.file3.setStyleSheet(
            "QPushButton{background-color:rgb(111,180,219)}"  # 按键背景色
            "QPushButton:hover{color:green}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )

        # 设置显示窗口参数
        self.fileT = QtWidgets.QPushButton(self)
        self.fileT1 = QtWidgets.QPushButton(self)
        self.fileT2 = QtWidgets.QPushButton(self)
        self.fileT3 = QtWidgets.QPushButton(self)
        self.fileT.setGeometry(QtCore.QRect(300, 50, 700, 28))
        self.fileT1.setGeometry(QtCore.QRect(300, 150, 700, 28))
        self.fileT2.setGeometry(QtCore.QRect(300, 250, 700, 28))
        self.fileT3.setGeometry(QtCore.QRect(300, 350, 700, 28))
        self.fileT.setObjectName("file")
        self.fileT1.setObjectName("file")
        self.fileT2.setObjectName("file")
        self.fileT3.setObjectName("file")
        self.fileT.setStyleSheet("background-color:rgb(111,180,219)")
        self.fileT1.setStyleSheet("background-color:rgb(111,180,219)")
        self.fileT2.setStyleSheet("background-color:rgb(111,180,219)")
        self.fileT3.setStyleSheet("background-color:rgb(111,180,219)")
        self.fileT.setStyleSheet(
            "QPushButton{background-color:rgb(111,180,219)}"  # 按键背景色
            "QPushButton:hover{color:green}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )
        self.fileT1.setStyleSheet(
            "QPushButton{background-color:rgb(111,180,219)}"  # 按键背景色
            "QPushButton:hover{color:green}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )
        self.fileT2.setStyleSheet(
            "QPushButton{background-color:rgb(111,180,219)}"  # 按键背景色
            "QPushButton:hover{color:green}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )
        self.fileT3.setStyleSheet(
            "QPushButton{background-color:rgb(111,180,219)}"  # 按键背景色
            "QPushButton:hover{color:green}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        )

        # 设置单选框
        rbtn1 = QRadioButton(self)
        rbtn2 = QRadioButton(self)
        btngroup1 = QButtonGroup(self)
        btngroup1.addButton(rbtn1)
        btngroup1.addButton(rbtn2)
        rbtn1.setText("迁移学习方法")
        rbtn1.move(50, 450)
        rbtn2.setText("深度学习方法")
        rbtn2.move(200, 450)

        #  设置输入框
        self.lb1 = QLabel('学习率：', self)
        self.lb1.move(50, 550)

        self.lb2 = QLabel('迭代次数：', self)
        self.lb2.move(250, 550)

        self.lb3 = QLabel('源域信号种类数：', self)
        self.lb3.move(500, 550)

        self.lb4 = QLabel('epoch：', self)
        self.lb4.move(820, 550)

        self.lb5 = QLabel(' 0.02 ', self)  # 学习率默认值为0.02
        self.lb5.move(100, 550)

        self.lb6 = QLabel(' 200 ', self)  # 迭代次数默认为2000
        self.lb6.move(320, 550)

        self.lb7 = QLabel(' 9 ', self)  # 源域种类默认9种，根据数据集选择
        self.lb7.move(600, 550)

        self.lb8 = QLabel(' 50 ', self)  # epoch共50轮
        self.lb8.move(870, 550)

        # 输出绘图框
        # chart = QChartView(self)
        # chart.setGeometry(QtCore.QRect(0, 600, 480, 300))
        # chart.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        # # chart._chart = QChart(title="折线图堆叠")  # 创建折线视图
        # chart.setBackgroundBrush(QBrush(QColor("#FFFFFF")))  # 改变图背景色


        # y_Aix = QValueAxis()  # 定义y轴
        # y_Aix.setLabelFormat("%d")
        # y_Aix.setRange(0, 1.0)
        # y_Aix.setTickCount(0.1)
        # chart.addAxis(y_Aix, Qt.AlignLeft)  # 添加到左侧
        # self.customAxisX(chart)

        # #  图形项默认无法接收悬停事件，可以使用QGraphicsItem的setAcceptHoverEvents()函数使图形项可以接收悬停事件。
        # chart._chart.setAcceptHoverEvents(True)
        # chart.createDefaultAxes()  # 创建默认的轴
        # chart.axisY().setTickCount(11)  # y1轴设置10个刻度
        # chart._chart.axisY().setLabelFormat("%d")
        # chart._chart.axisY().setRange(100, 200)  # 设置y1轴范围

        # 设置按键
        self.bt1 = QPushButton('修改学习率', self)
        self.bt1.move(150, 545)

        self.bt2 = QPushButton('修改迭代次数', self)
        self.bt2.move(380, 545)

        self.bt3 = QPushButton('修改源域信号种类数', self)
        self.bt3.move(650, 545)

        self.bt4 = QPushButton('修改epoch数', self)
        self.bt4.move(920, 545)

        self.train_bt = QPushButton('开始训练', self)
        timer = QtCore.QTimer()
        self.timer = timer

        # 输出文本框
        self.textlabel = QLabel('↓结果输出框图↓', self)
        self.textlabel.setGeometry(QtCore.QRect(500, 600, 100, 28))
        self.view_area = QtWidgets.QPlainTextEdit(self)
        self.view_area.setGeometry(QtCore.QRect(200, 650, 680, 250))

        # self.train.move(600, 450)
        self.train_bt.setGeometry(QtCore.QRect(600, 450, 200, 30))  # （横坐标，纵坐标，长，宽），原点为左上角
        self.train_bt.setStyleSheet(
            "QPushButton{background-color:rgb(255,255,224)}"  # 按键背景色
            "QPushButton:hover{color:green}"  # 光标移动到上面后的前景色
            "QPushButton{border-radius:6px}"  # 圆角半径
            "QPushButton:pressed{background-color:rgb(180,0,0);border: None;}"  # 按下时的样式
        )

        self.show()

        # 设置按键功能
        self.bt1.clicked.connect(self.showDialog)
        self.bt2.clicked.connect(self.showDialog)
        self.bt3.clicked.connect(self.showDialog)
        self.bt4.clicked.connect(self.showDialog)

        self.file.clicked.connect(self.msg)
        self.file1.clicked.connect(self.msg)
        self.file2.clicked.connect(self.msg)
        self.file3.clicked.connect(self.msg)

        self.train_bt.clicked.connect(self.training)

    def showDialog(self):
        sender = self.sender()
        """
        若我们按下按钮1，此时显示输入对话框。 第一个字符串是一个对话标题，第二个是对话框中的一个消息。 对话框返回输入的文本和布尔值。 如果我们点击Ok按钮，布尔值为true。
        """
        if sender == self.bt1:
            text, ok = QInputDialog.getText(self, '修改学习率', '请输入学习率：')
            if ok:
                self.lb5.setText(str(text))  # 如果我们按下ok键，则标签4的text值是从对话框接收的文本。
        elif sender == self.bt2:
            text, ok = QInputDialog.getText(self, '修改迭代次数', '请输入迭代次数：')  # 可以输入整数，最小值、最大值可以自己设定。
            if ok:
                self.lb6.setText(str(text))
        elif sender == self.bt3:
            text, ok = QInputDialog.getText(self, '修改源域信号种类数', '请输入源域信号种类数: ')  # 输入种类数
            if ok:
                self.lb7.setText(str(text))
        elif sender == self.bt4:
            text, ok = QInputDialog.getText(self, '修改epoch数', '请输入epoch数: ')  # 输入epoch数
            if ok:
                self.lb8.setText(str(text))

    # 文件夹选择功能
    def msg(self, Filepath):
        sender = self.sender()
        if sender == self.file:
            m, ok = QtWidgets.QFileDialog.getOpenFileName(None,
                                                          "选取单个文件", "C:/", "All Files (*);;Text Files (*.txt)")
            if ok:
                self.fileT.setText(m)
        elif sender == self.file1:
            m, ok = QtWidgets.QFileDialog.getOpenFileName(None,
                                                          "选取单个文件", "C:/", "All Files (*);;Text Files (*.txt)")
            if ok:
                self.fileT1.setText(m)
        elif sender == self.file2:
            m = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "C:/")  # 起始路径
            self.fileT2.setText(m)
        elif sender == self.file3:
            m = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "C:/")  # 起始路径
            self.fileT3.setText(m)

    def write(self, text):
        """"""
        cursor = self.view_area.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text + "\n")
        self.view_area.setTextCursor(cursor)
        self.view_area.ensureCursorVisible()

    def bind_trigger(self):
        # Button clicked trigger
        self.button.clicked.connect(lambda: self.timer.start(1))
        self.timer.timeout.connect(self.clicked_button)

    def clicked_button(self):
        self.write("Write Num: {0}".format(self.timer.interval()))
        self.timer.setInterval(self.timer.interval() + 1)

    def training(self):
        cursor = self.text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.text.setTextCursor(cursor)
        self.view_area.ensureCursorVisible()


        self.train_bt.setText(str('训练中...'))  # 如果我们按下ok键，则标签4的text值是从对话框接收的文本。
        trainpath = str(self.fileT)
        testpath = str(self.fileT1)
        model_save_path = str(self.fileT2)
        trainlog_save_path = str(self.fileT3)

        learning_rate = str(self.lb5)
        batch_size = str(self.lb6)
        class_num = str(self.lb7)
        epoch_num = str(self.lb8)





if __name__ == '__main__':
    app = QApplication(sys.argv)
    MW = My_Window()
    sys.exit(app.exec_())










    # # 自定义x轴(均分)
    # def customAxisX(self, chart):
    #     chart = chart
    #     series = chart.series()
    #     if not series:
    #         return
    #     # 获取当前时间前8小时的一小时内的时间
    #     time = []
    #     for index in range(13):
    #         num = 60 / 13
    #         last_day = (datetime.datetime.now() + datetime.timedelta(hours=-8, minutes=- index * num)).strftime(
    #             "%H:%M")
    #         time.append(last_day)
    #     category = list(reversed(time))
    #
    #     '''QValueAxis是轴的范围什么的不需要自己指定，轴上显示的label（也就是0,1,2,3这些内容）是默认的。
    #     qt会根据你轴上的点自动设置。若你需要自定义一些内容，QCategoryAxis是比较好的，但是需要自己自定义好才可以调用。'''
    #     axisx = QCategoryAxis(
    #         chart, labelsPosition=QCategoryAxis.AxisLabelsPositionOnValue)
    #
    #     axisx.setGridLineVisible(False)  # 隐藏网格线条
    #     axisx.setTickCount(len(category))  # 设置刻度个数
    #     minx = chart.axisX().min()
    #     maxx = chart.axisX().max()
    #     tickc = chart.axisX().tickCount()
    #     print(tickc)
    #     if tickc < 2:
    #         axisx.append(category[0])
    #     else:
    #         step = (maxx - minx) / (tickc - 1)  # tickc>=2
    #         for i in range(0, tickc):
    #             axisx.append(category[i], minx + i * step)
    #             # 保存x轴值
    #     chart.setAxisX(axisx, series[-1])

