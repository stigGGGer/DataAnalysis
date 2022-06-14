# надо для графиков
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
#import matplotlib.pyplot as plt

def Draw2DGraph(form,x,y,color):
   form.figure.clear() # чистим график
   ax = form.figure.add_subplot() # надо
   scatter = ax.scatter(x,y,c = color,cmap='rainbow') # задаем точки
   ax.set_xlabel(x.name) # задаем название осей
   ax.set_ylabel(y.name)
   ax.legend(*scatter.legend_elements(), title="Classes")
   form.canvas.draw()  # обновляем отрисовку графика

def Draw3DGraph(form,x,y,z,color):
    form.figure.clear() # чистим график
    ax = form.figure.add_subplot(projection='3d') # надо
    form.figure.tight_layout() # увеличиваем размер графика
    scatter = ax.scatter(x,y,z,c = color,cmap='rainbow') # задаем точки
    ax.set_xlabel(x.name) # задаем название осей
    ax.set_ylabel(y.name)
    ax.set_zlabel(z.name)
    ax.legend(*scatter.legend_elements(), title="Classes")
    form.canvas.draw() # обновляем отрисовку графика
