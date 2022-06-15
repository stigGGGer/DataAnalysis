from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import time
import sys

from PyQt5.QtCore import QThread, pyqtSignal

import threading
import numpy as np

# метрики
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,mean_absolute_error

# алгоритмы и средство визуализации
from classification.k_nearest_neighbors import *
from clustering.agglomerative_clustering import *
from clustering.Affinity_Propagation import *
from clustering.Optics import *
from clustering.K_Means import *
from classification.mySGDClassifier import *
from classification.SVM import *
from chart.tools import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# для окон ошибок
from PyQt5.QtWidgets import QMessageBox

# для парсинга пути к файлу
from pathlib import Path

# dataframe
import pandas as pd

# надо для графиков
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

Form, _ = uic.loadUiType("form.ui")


class Calculate(QThread):
    mySignal = pyqtSignal(str)                       
    
    def __init__(self) -> None:
        super().__init__()
        self.table = None
        self.D = None
        self.algorithm = None
        self.target = None
        self.metricsTrue = None

        self.result = None
        self.c = None
        self.names = None
        self.parametrs = None

    def run(self):
        try:
          #  print("123")
            data = []
            result = None
            if self.algorithm == "Nearest_Neighbors":
                 result = Nearest_Neighbors(self.table,self.target,self.parametrs)
            elif self.algorithm == "Agglomerative":
                 result = Agglomerative_Clustering(self.table,self.parametrs)
                 #self.result = result[0]
                 #self.c = result[1]
            
            # Великов - кластеризация     
            elif self.algorithm == "Affinity_Propagation":
                result = Affinity_Propagation(self.table, self.parametrs)

            ## Великов - классификация
            elif self.algorithm == "SVM":
                result = mySVM(self.table,self.target,self.parametrs)

            # Чугунов - кластеризация
            elif self.algorithm == "OPTICS":
                result = Optics(self.table, self.parametrs)

            ## Чугунов/Гуляев - классификация
            elif self.algorithm == "SGDClassifier":
                result = mySGDClassifier(self.table,self.target,self.parametrs)

            ## Гуляев - кластеризация
            elif self.algorithm == "K_Means":
                result = K_Means(self.table, self.parametrs)

            #govno generators co. (c) 2022

            self.result = result[0]
            self.c = result[1]


            if len(result)>2:
                self.metricsTrue = result[2]
            else:
                self.metricsTrue = self.target
          #  print("222")
            self.mySignal.emit('Success')                    
        except Exception as err:
            self.mySignal.emit(str(err))                          

class Canvas(QMainWindow):
    def __init__(self):
        super(Canvas, self).__init__()
        self.ui = Form()
        self.ui.setupUi(self)        

        self.thread = Calculate()
        self.canvas = self.ui
        self.thread.mySignal.connect(self.Result)
        self.ConfigureInterface()

    def OpenFile(self):
        self.FillAlgorithm(False)
        path = QFileDialog.getOpenFileName(self,  'Выберите файл',directory = "./data/",filter = "Excel (*.csv *.xlsx)")[0]
        
        # если отменили открытие файла
        if path=="":
            return

        try:# sep=";"
          self.dataset = pd.read_csv(path,sep=";",encoding = "ISO-8859-1")
          #self.dataset = self.dataset.reset_index()
         # print(self.dataset.head(6))
          self.ui.label_5.setText(Path(path).name)
          self.ui.textEdit_2.setText(path)
          self.FillAlgorithm(True)
        except Exception as err:
          self.ui.label_5.setText("")
          self.ui.textEdit_2.setText("")   
          self.FillAlgorithm(False)
          QMessageBox.about(self, "Ошибка!", "Не удалось открыть файл!\n"+str(err))
    

    def SwitchAlgorithm(self):
        self.ui.comboBox_2.clear()
        type = self.ui.comboBox.currentText()
        if type == "Классификация":
            self.ui.comboBox_2.addItem('Nearest_Neighbors')
            self.ui.comboBox_2.addItem('SGDClassifier')
            self.ui.comboBox_2.addItem('SVM')            
           # self.ui.comboBox_3.show()
            #self.ui.label_9.show()
        elif type == "Кластеризация":
            self.ui.comboBox_2.addItem('Agglomerative')
            self.ui.comboBox_2.addItem('Affinity_Propagation')
            self.ui.comboBox_2.addItem('OPTICS')
            self.ui.comboBox_2.addItem('K_Means')
            #self.ui.comboBox_3.hide()
           # self.ui.label_9.hide()

        #self.ui.comboBox_3.clear()
        self.ui.comboBox_4.clear()
        self.ui.comboBox_5.clear()
        self.ui.comboBox_6.clear()
        self.SwitchParametrs()

    def FillXYZT(self):
        x = self.ui.comboBox_5.currentText()
        y = self.ui.comboBox_4.currentText()
        z = self.ui.comboBox_6.currentText()
        t = self.ui.comboBox_3.currentText()

        self.ui.comboBox_3.clear()
        self.ui.comboBox_4.clear()
        self.ui.comboBox_5.clear()
        self.ui.comboBox_6.clear()

        self.ui.comboBox_5.addItem(x)
        self.ui.comboBox_4.addItem(y)
        self.ui.comboBox_6.addItem(z)
        self.ui.comboBox_3.addItem(t)

        if x!="":
            self.ui.comboBox_5.addItem("")
        if y!="":
            self.ui.comboBox_4.addItem("")
        if z!="":
            self.ui.comboBox_6.addItem("")
        if t!="":
            self.ui.comboBox_3.addItem("")

        for name in self.dataset.columns:
            if name != x and name != y and name != z and name != t:
               self.ui.comboBox_3.addItem(name)
               self.ui.comboBox_4.addItem(name)
               self.ui.comboBox_5.addItem(name)
               self.ui.comboBox_6.addItem(name)

        self.flag = False



    def FillAlgorithm(self,flag):
        if flag==True:
            self.ui.comboBox.addItem('Классификация')
            self.ui.comboBox.addItem('Кластеризация')    
            self.SwitchAlgorithm()
            self.ui.buttonLaunch.show()
            #self.ui.buttonCopyToExcel.show()
        else:
            self.ui.comboBox.clear()
            self.ui.comboBox_2.clear()
            self.ui.comboBox_3.clear()
            self.ui.comboBox_4.clear()
            self.ui.comboBox_5.clear()
            self.ui.comboBox_6.clear()
            self.ui.buttonLaunch.hide()
           # self.ui.buttonCopyToExcel.hide()
            self.SwitchParametrs()


    def SwitchParametrs(self):
        algorithm = self.ui.comboBox_2.currentText()
        self.thread.algorithm = algorithm

       # self.ui.widget123.hide()

        if algorithm == "":
          # for widget in self.ui.tab.children():
           #    widget.hide()

           self.ui.widget.hide()
           ###
           #self.ui.tab.layout = QVBoxLayout() 
           #self.ui.tab.setLayout(self.ui.tab.layout)
      
        else:
            self.FillXYZT()
            for i in reversed(range(self.ui.tab.layout.count())): 
                widgetToRemove = self.ui.tab.layout.itemAt(i).widget()
                self.ui.tab.layout.removeWidget(widgetToRemove) ##??? иначе может ошибка возникнуть
                widgetToRemove.setParent(self.ui.widget)

            for widget in self.ui.widget.children():            
                if algorithm == widget.objectName():
                     self.ui.tab.layout.addWidget(widget)
                     self.ui.tab.setLayout(self.ui.tab.layout)
                     break

        self.ui.textEdit.setText("Ожидаю...")
        self.textEdit123.setText("")
        self.figure.clear()
        self.canvas.draw()
        self.ui.buttonCopyToExcel.hide()
        self.ui.tableWidget.setColumnCount(0)
        self.ui.tableWidget.setRowCount(0)
 

    def Result(self, result):
            try:
               if result == 'Success':
                #self.ui.textEdit.setText("Успех!")
                #self.ui.buttonLaunch.setText("Запустить")
                self.SwitchEnabled(True)
                self.ui.textEdit.setText("Успех!") 
                self.ui.buttonCopyToExcel.show()
                columnsCount = 0
                rowsCount = len(self.thread.c)

                if self.thread.D == "2D":
                    Draw2DGraph(self,self.thread.result.iloc[:, 0],self.thread.result.iloc[:, 1],self.thread.c)
                    columnsCount = 2
                elif self.thread.D == "3D":
                    Draw3DGraph(self,self.thread.result.iloc[:, 0],self.thread.result.iloc[:, 1],self.thread.result.iloc[:, 2],self.thread.c)
                    columnsCount = 3
                
           #     rowsCount = len(self.thread.c)
                self.ui.tableWidget.setColumnCount(columnsCount+1)
                self.ui.tableWidget.setRowCount(rowsCount)

           #     names = []
           #     names.append()
                names =  list(self.thread.result.columns.values)
                names.append("cluster")
                self.ui.tableWidget.setHorizontalHeaderLabels(names) 
                self.thread.names = names

                for i in range(rowsCount):
                    self.ui.tableWidget.setItem(i, columnsCount, QTableWidgetItem(str(self.thread.c[i])))
                    for j in range(columnsCount):
                        self.ui.tableWidget.setItem(i, j, QTableWidgetItem(str(self.thread.result.iat[i,j])))
               
                self.CalculateMetrics()
               else:
                  raise Exception(result)
               
            except Exception as err:

            #self.thread.table["color"] = self.thread.c
            #print(self.thread.result.iat[0,0])
           # for i, row in self.thread.result.iterrows():
           #  self.ui.tableWidget.setRowCount(self.ui.tableWidget.rowCount() + 1)
           #
           #  for j in range(self.ui.tableWidget.columnCount()-1):
           #      self.ui.tableWidget.setItem(i, j, QTableWidgetItem(str(row[j])))


           # for i in range(length):
           #      self.ui.tableWidget.setItem(i, j, QTableWidgetItem(str(row[j])))
           #
           # table.setHorizontalHeaderLabels(headers)
           # self.ui.tableWidget.setRowCount(100)
           # self.ui.tableWidget.setItem(0,0,QTableWidgetItem("123"))
           # self.ui.tableWidget.resizeColumnsToContents()
           # self.ui.tableWidget.resizeRowsToContents()
           # self.ui.tableWidget.setHorizontalHeaderItem(0,QTableWidgetItem("XX"))

          #  QMessageBox.information(self, 
          #      'Оповещение', 'Файл преобразован и успешно сохранен!')

        
              self.figure.clear()
              self.canvas.draw()
              self.figure2.clear()
              self.canvas2.draw()
              self.textEdit123.setText("")
              self.ui.tableWidget.setColumnCount(0)
              self.ui.tableWidget.setRowCount(0)
              self.SwitchEnabled(True)
              self.ui.buttonCopyToExcel.hide()
              self.ui.textEdit.setText("Ошибка вычислений:\n"+str(err)) 


           # QMessageBox.critical(self, 'Ошибка', 'Что-то пошло не так')


    def CalculateMetrics(self):
        list = []
        list.clear()
       
        mat_con = confusion_matrix(self.thread.metricsTrue, self.thread.c)
        list.append("Accuracy: "+str(round(accuracy_score(self.thread.metricsTrue, self.thread.c),3)))
        list.append("Precision: "+str(round(precision_score(self.thread.metricsTrue, self.thread.c, average='weighted',zero_division=0),3)))
        list.append("Recall: "+str(round(recall_score(self.thread.metricsTrue, self.thread.c, average='weighted',zero_division=0),3)))
        list.append("Mean absolute error: "+str(round(mean_absolute_error(self.thread.metricsTrue, self.thread.c),3)))
        list.append("F1: "+str(round(f1_score(self.thread.metricsTrue, self.thread.c, average='weighted',zero_division=0),3)))
       # list.append("666")
        self.figure2.clear() # чистим график
       # mat_con = confusion_matrix([2,0,2,2,2,0], [0,2,2,0,0,2])
        ax = self.figure2.add_subplot() # надо
        ax.set_title("Матрица неточности")
       # scatter = ax.scatter([1,2,3],[1,2,3],c = [1,2,3],cmap='rainbow') # задаем точки
        ax.set_xlabel("Predictions") # задаем название осей
        ax.set_ylabel("Actuals")
       # plt.xlabel('Predictions', fontsize=16)
       # plt.ylabel('Actuals', fontsize=16)
        ax.matshow(mat_con, cmap=plt.cm.YlOrRd, alpha=0.5)
        for m in range(mat_con.shape[0]):
            for n in range(mat_con.shape[1]):
                ax.text(x=m,y=n,s=mat_con[m, n], va='center', ha='center', size='xx-large')
        #ax.legend(*scatter.legend_elements(), title="Classes")
        self.canvas2.draw()  # обновляем отрисовку графика
        #self.thread.target.drop_duplicates()
        #print("Confusion Matrix: \n"+str(confusion_matrix(y_true, y_pred,labels=[2,0])))
        #print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
        #print("Precision: " + str(precision_score(y_true, y_pred,  average='weighted')))
        #print("Recall: " + str(recall_score(y_true, y_pred,  average='weighted')))
        #print("Mean absolute error: " + str(mean_absolute_error(y_true, y_pred)))
        #print("F1: " + str(f1_score(y_true, y_pred, average='weighted')))
        self.textEdit123.setText('\n'.join(list))

    def SwitchEnabled(self,flag):
         self.ui.comboBox.setEnabled(flag)
         self.ui.comboBox_2.setEnabled(flag)
         self.ui.comboBox_3.setEnabled(flag)
         self.ui.comboBox_4.setEnabled(flag)
         self.ui.comboBox_5.setEnabled(flag)
         self.ui.comboBox_6.setEnabled(flag)
         self.ui.buttonOpenFile.setEnabled(flag)
        # self.ui.buttonCopyToExcel.setEnabled(flag)    
         
         if flag:
            self.ui.textEdit.setText("Ожидаю...")
            self.ui.buttonLaunch.setText("Запустить")
            self.figure.clear()
            self.canvas.draw()
            self.figure2.clear()
            self.canvas2.draw()
            self.textEdit123.setText("")
            self.ui.tableWidget.setColumnCount(0)
            self.ui.tableWidget.setRowCount(0)
            self.ui.buttonCopyToExcel.hide()
         else:
            self.ui.textEdit.setText("Происходят вычисления...")
            self.ui.buttonLaunch.setText("Остановить")
            self.ui.buttonCopyToExcel.hide()


          
    def Launch(self):  
                if  self.thread.isRunning():
                    self.thread.terminate()
                    self.SwitchEnabled(True)
                   # self.ui.textEdit.setText("Ожидаю...")
                   # self.ui.buttonLaunch.setText("Запустить")
                    return
                if self.ui.comboBox_5.currentText()=="" or self.ui.comboBox_4.currentText()=="":
                    QMessageBox.about(self, "Ошибка!", "Поля X и Y не выбраны!")
                    return

                if  self.ui.comboBox_3.currentText() == "":
                     QMessageBox.about(self, "Ошибка!", "Поле Target не выбрано!")
                     return
                

                #Создаем таблицу
                if self.flag == False:
                    try:

                    # print("11111111111111111")
                     #data = np.array([[1,2,8,9],[3,4,5,6]])
                     #self.thread.table = pd.DataFrame(data)
                     #print(str(len(self.thread.table.columns)))
                     #data.clear()
                     data = []
                     data.clear()

                    #df = pd.DataFrame()
                    #df.columns = ["lala","bbbb"]
                    # data.clear()
                     data.append(self.ui.comboBox_5.currentText())
                     data.append(self.ui.comboBox_4.currentText())

                     if self.ui.comboBox_6.currentText() != "": # z
                         data.append(self.ui.comboBox_6.currentText())
                         self.thread.D = "3D"
                     else:
                         self.thread.D = "2D"


                     

                     #if self.ui.comboBox.currentText()=="Классификация": # target
                     self.thread.target = self.dataset.iloc[:,self.dataset.columns.get_loc(self.ui.comboBox_3.currentText())]

                         # находим уникальные значения
                     df = self.thread.target.drop_duplicates()
                     #df2 = df.copy()
                     ##for i in df.items():
                     #for i in range(len(df)):
                     #   df2.iat[i] = i
               
                        # print(self.thread.target.head(5))
                        # a= self.thread.target.iat[0]# = "123"
                        # print("123123")
                         # заменяем их на числа
                     a = 0
                     for i in self.thread.target.items():
                         for j in df.items():
                            if i[1]==j[1]:
                               self.thread.target.iat[i[0]] = a
                               break
                            a=a+1
                         a = 0
                     self.thread.target = pd.to_numeric(self.thread.target)
                       # print(df.head(10))
                         # переводим значения в числа
                         #items = []
                         #for data in df.items():
                         #    for i in items:
                         #        if data!=i:

                         #self.dataset.columns.get_loc(self.dataset[self.ui.comboBox_6.currentText())

                  #   a = self.dataset.columns.get_loc(self.ui.comboBox_3.currentText())
                    # self.thread.table = data
                     self.thread.table = self.dataset[data]
                     self.flag = True
                     #print("2222222222222222222222222")
                    # print(123)
                    # self.thread.table = self.thread.table.transpose()
                  #  print(self.thread.table.head(5))
                  #  print(str(len(self.thread.table.columns)))
                  #  print(str(len(self.thread.target.columns)))
               # se lf.ui.textEdit.setText("Происходят вычисления...") 
                    except Exception as err:      
                         QMessageBox.about(self, "Ошибка!", "Ошибка считывания данных выбранных столбцов.\n"+str(err))
                         return

                self.thread.parametrs = self.SetParametrs()
               # self.ui.buttonLaunch.setText("Остановить")
                self.SwitchEnabled(False)
                self.thread.start()
                
                #self.btn.clicked.connect(self.bc.start)
               # threading.Thread(target = self.Algorithm,args=(self,)).start()

    def SetParametrs(self):
       parametrs = []

       #spinBox .value()
       #comboBox .currentText()

       if self.thread.algorithm == "Nearest_Neighbors":
           parametrs.append(self.ui.spinBox.value())
           parametrs.append(self.ui.comboBox_7.currentText())
           parametrs.append(self.ui.comboBox_8.currentText())
           parametrs.append(self.ui.doubleSpinBox.value())
       elif self.thread.algorithm == "Agglomerative":
           parametrs.append(self.ui.comboBox_9.currentText())
           parametrs.append(self.ui.spinBox_2.value())
           parametrs.append(self.ui.comboBox_10.currentText())
       elif self.thread.algorithm == "Affinity_Propagation":
           parametrs.append(self.ui.comboBox_11.currentText())
           parametrs.append(self.ui.spinBox_3.value())
           parametrs.append(self.ui.doubleSpinBox_2.value())
           parametrs.append(self.ui.doubleSpinBox_3.value())
       elif self.thread.algorithm == "OPTICS":
           parametrs.append(self.ui.spinBox_4.value())
           parametrs.append(self.ui.doubleSpinBox_4.value())
           parametrs.append(self.ui.doubleSpinBox_5.value())
       elif self.thread.algorithm == "K_Means":
           parametrs.append(self.ui.spinBox_5.value())
           parametrs.append(self.ui.comboBox_12.currentText())
           parametrs.append(self.ui.spinBox_6.value())
       elif self.thread.algorithm == "SGDClassifier":
           parametrs.append(self.ui.doubleSpinBox_6.value())
           parametrs.append(self.ui.spinBox_7.value())
           parametrs.append(self.ui.doubleSpinBox_7.value())
       elif self.thread.algorithm == "SVM":            
           parametrs.append(self.ui.doubleSpinBox_8.value())
           parametrs.append(self.ui.comboBox_13.currentText())

       return parametrs 

    def CopyToExcel(self):
         path = QFileDialog.getSaveFileName(self, 'Выберите файл',filter = "Excel (*.csv)")[0]

         if path!="":
             try:
                df = pd.DataFrame(columns = self.thread.names)
                
                for row in range(self.ui.tableWidget.rowCount()):
                   for col in range(self.ui.tableWidget.columnCount()):
                     df.at[row,self.thread.names[col]] = self.ui.tableWidget.item(row,col).text()
                df.to_csv(path , index = False)
                QMessageBox.about(self, "Уведомление!", "Файл успешно сохранен!")
             except Exception as err:
                QMessageBox.about(self, "Ошибка!", "Сохранить файл не удалось!\n"+str(err))  

    

    def ShowReference(self):
        QMessageBox.about(self, "Руководство пользователя", "Приложение позволяет выполнять кластеризацию и классификацию.\nДля запуска алгоритмов потребуется:\n1) Выбрать файл формата .csv или .xlsx\n2) Выбрать тип моделирования: Классификация или Кластеризация\n3) Выбрать алгоритм для вычислений\n4) Ввести параметры для алгоритма\n5) Нажать кнопку запуска")

    def ConfigureInterface(self):
        #self.setFixedSize(966, 622)  # фиксиурем размер окна

        self.ui.comboBox.activated.connect(self.SwitchAlgorithm)
        self.ui.comboBox_2.activated.connect(self.SwitchParametrs)
        self.ui.comboBox_3.activated.connect(self.FillXYZT)
        self.ui.comboBox_4.activated.connect(self.FillXYZT)
        self.ui.comboBox_5.activated.connect(self.FillXYZT)
        self.ui.comboBox_6.activated.connect(self.FillXYZT)
        self.ui.buttonOpenFile.clicked.connect(self.OpenFile)
        self.ui.buttonLaunch.clicked.connect(self.Launch)
        self.ui.buttonCopyToExcel.clicked.connect(self.CopyToExcel)
        self.ui.buttonReference.clicked.connect(self.ShowReference)
        #self.ui.textEdit.textChanged.connect(self.table)
            
        self.flag = False

        for i in AffinityType:
            self.ui.comboBox_10.addItem(i.name)

        for i in LinkageType:
            self.ui.comboBox_9.addItem(i.name)

        for i in KNNAlgorithmType:
            self.ui.comboBox_7.addItem(i.name)

        for i in KNNWeightType:
            self.ui.comboBox_8.addItem(i.name)

        for i in Affinity:
            self.ui.comboBox_11.addItem(i.name)

        for i in covariance_type:
            self.ui.comboBox_12.addItem(i.name)

        for i in Gamma:
            self.ui.comboBox_13.addItem(i.name)

        # нужно чтобы добавлять элементы во вкладку
        self.ui.tab_2.layout = QVBoxLayout() 

        # сам график
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # разные кнопки взаимодействия с графиком
        self.toolbar = NavigationToolbar(self.canvas, self)

        # добавляем все на вкладку
        self.ui.tab_2.layout.addWidget(self.toolbar)
        self.ui.tab_2.layout.addWidget(self.canvas)
        self.ui.tab_2.setLayout(self.ui.tab_2.layout)

        self.ui.tab_4.layout = QVBoxLayout() 
        self.figure2 = plt.figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.textEdit123 = QTextEdit()
        self.textEdit123.setReadOnly(True)
        self.ui.tab_4.layout.addWidget(self.canvas2)
        self.ui.tab_4.layout.addWidget(self.textEdit123)
        self.ui.tab_4.setLayout(self.ui.tab_4.layout)

        self.ui.buttonLaunch.hide()
        self.ui.buttonCopyToExcel.hide()
        self.ui.tab.layout = QVBoxLayout() 
        self.SwitchParametrs()


if __name__ == "__main__":
    app = QApplication([])
    app.setStyle('Breeze')

    application = Canvas()    
    app2 = application
    application.show()
    sys.exit(app.exec())
