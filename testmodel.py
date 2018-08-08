import numpy as np
from sklearn.svm import SVC
import pickle
import loaddata
import extractfeature

label_map_20 = {0: 'bear',
                1: 'bird',
                2: 'car',
                3: 'cow',
                4: 'elk',
                5: 'fox',
                6: 'giraffe',
                7: 'horse',
                8: 'koala',
                9: 'lion',
                10: 'monkey',
                11: 'plane',
                12: 'puppy',
                13: 'sheep',
                14: 'statue',
                15: 'tiger',
                16: 'tower',
                17: 'train',
                18: 'whale',
                19: 'zebra',
                20: 'bicycle'}


def testmodel(input_path, model_path='group_6.model', top=5):

    data,tag = loaddata.loaddata(input_path)
    print("test_list")
    input_img = extractfeature.extractfeature(data,tag)
    print("test_feature")

    result = []
    img_data = []
    img_tag = []
    with open(model_path, 'r') as f:
        clf = pickle.load(f)
    for img in input_img:
        img_data.append(img[0])
        img_tag.append(int(img[1]))
    predict_prob = clf.predict_log_proba(img_data)

    for idx, score in enumerate(predict_prob):
        sorted_index = np.argsort(-score).tolist()
        rank = sorted_index.index(img_tag[idx])
        if rank < top:
            result.append("True")
        else:
            result.append("False")
    print('Top-5: ')
    print(result.count("True")*1.0/len(result))
    return result
