import numpy as np
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import GridSearchCV


def trainmodel(input_img, out_path='group_6.model', in_C=0.8, in_kernel='linear'):
    clf = SVC(C=in_C, kernel=in_kernel, probability=True)
    img_data = []
    img_tag = []
    i=0
    for img in input_img:
        i+=1
        img_data.append(img[0])
        img_tag.append(int(img[1]))
        # print(img[0])
        print(i)
    clf.fit(img_data, img_tag)
    with open(out_path, 'w') as f:
        pickle.dump(clf, f)
