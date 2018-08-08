import extractfeature
import loaddata
import trainmodel
import testmodel

if __name__ == '__main__':
    train_path = "/home/fanfan/practice/ds2018"
    test_path = "/home/fanfan/practice/test1"
    train_data,train_tag = loaddata.loaddata(train_path)
    print("train_list")
    train_feature = extractfeature.extractfeature(train_data,train_tag)
    print("train_feature")
    trainmodel.trainmodel(train_feature)
    print("train finished")
    testmodel.testmodel(test_path)






    # data,tag = loaddata.loaddata(test_path)
    # print("test_list")
    # test_feature = extractfeature.extractfeature(data,tag)
    # print("test_feature")