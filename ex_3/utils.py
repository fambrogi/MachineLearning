import matplotlib.pyplot as plt
import itertools
import numpy as np

def printConfusionMatrix(matrix, ds, title='', what='') :

    fs = 12
    plt.figure(figsize=(6, 5))
    # place labels at the top
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    # plot the matrix per se
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    # plot colorbar to the right
    plt.colorbar()
    fmt = 'd'
    fmt = 'f.2'
    # write the number of predictions in each bucket
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, '{0:.2f}'.format(matrix[i, j]), horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    # all the same in this case
    ticks = {'income'  : ['0', '1'],
             'titanic': ['0', '1'],
             'social' : ['0', '1']}

    #classes = ticks[dataset]

    classes = range(0, len(matrix))
    tick_marks = np.arange(len(classes))

    try:
        plt.xticks(tick_marks, ticks[ds], rotation=20, fontsize=8)
        plt.yticks(tick_marks, ticks[ds], rotation=20, va="center", fontsize=8)
    except:
        pass
    titles = {'income': 'Class',
              'titanic': 'Survive',
              'social': 'Buy'}

    plt.title( titles[ds] + ' for ' + what)

    plt.ylabel('True label', size=fs)
    plt.xlabel('Predicted label', size=fs)
    plt.tight_layout()
    plt.savefig('ConfusionMatrixes/' + ds + '_' + what + '_' + title + '.png', dpi=150)
    plt.close()

