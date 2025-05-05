#绘制测试的混淆矩阵图
import matplotlib.pyplot as plt
import numpy as np
# 绘制混淆矩阵函数
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.binary, mod_labels=[]):
    plt.figure(figsize=(10, 10), dpi=120)
    ind_array = np.arange(len(mod_labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c>=1e-5:
            plt.text(x_val, y_val, "%0.2f%s" % (c*100,'%'), color='red', fontsize=7, va='center', ha='center')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix Overall Snr")
    plt.colorbar()
    tick_marks = np.arange(len(mod_labels))
    plt.xticks(tick_marks, mod_labels, rotation=45)
    plt.yticks(tick_marks, mod_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(title+'.png', format='png', dpi=600)


def matadd(mat=None):
  res = 0
  for q in range(len(mat)):
      res+=mat[q][q]
  return res