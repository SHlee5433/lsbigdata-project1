from sklearn.metrics import confusion_matrix
import numpy as np
# 아델리: 'A'
# 친스트랩(아델리 아닌것): 'C'
y_true = np.array(['A', 'A', 'C', 'A', 'C', 'C', 'C'])
y_pred = np.array(['A', 'C', 'A', 'A', 'A', 'C', 'C'])
y_pred2 = np.array(['C', 'A', 'A', 'A', 'C', 'C', 'C'])

conf_mat = confusion_matrix(y_true=y_true, 
                 y_pred=y_pred,
                 labels=["A", "C"])

conf_mat

from sklearn.metrics import ConfusionMatrixDisplay

p = ConfusionMatrixDisplay(confusion_matrix = conf_mat,
                           display_labels = ("Adelie", "Chinstrap"))
p.plot(cmap = "Blues")

conf_mat = confusion_matrix(y_true=y_true, 
                 y_pred=y_pred2,
                 labels=["A", "C"])

conf_mat

from sklearn.metrics import ConfusionMatrixDisplay

p = ConfusionMatrixDisplay(confusion_matrix = conf_mat,
                           display_labels = ("Adelie", "Chinstrap"))
p.plot(cmap = "Blues")