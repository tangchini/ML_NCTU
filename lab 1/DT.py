## you cannot use sklearn
import glob
import numpy as np
import csv
from Function import *

def read_csv(filename):
    t = []
    csv_file = open(filename,'r')
    i =0
    for row in csv.reader(csv_file):
        # if (i > 2):
        t.append(row)
        # i+=1
    # print (t)
    t = np.array(t,dtype = np.float32)
    return t

def accF (x,y):
    acc = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            acc +=1
    acc = acc / len(x)
    return acc 

def feature_ex(x):
    t = []
    x = np.array(x)
    indexAx = 0
    indexAy = 1
    indexAz = 2
    indexGx = 3
    indexGy = 4
    indexGz = 5
    indexTotalAcc = 6
    indexTotalGyro = 7
    indexRoll = 8
    indexPitch = 9
    
    totalAcc = getTotalAxes(x[indexAx],x[indexAy],x[indexAz])
    totalGyro = getTotalAxes(x[indexGx],x[indexGy],x[indexGz])
    roll = getRoll(x[indexAx],x[indexAz])
    pitch = getPitch(x[indexAy],x[indexAz])

    processedKoalaData = np.ones((10,40))

    for i in range(6):
        processedKoalaData[i] = copy.deepcopy(x[i])
    processedKoalaData[6] = copy.deepcopy(totalAcc)
    processedKoalaData[7] = copy.deepcopy(totalGyro)
    processedKoalaData[8] = copy.deepcopy(roll)
    processedKoalaData[9] = copy.deepcopy(pitch)

    mean = getMean2D(processedKoalaData)
    t.append(mean)


    return t
 
# ##################################
#%%-------- load data
# ##################################

sensor = []
label = []
feature = []
f = glob.glob(r'40_data/down'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([0])

f = glob.glob(r'40_data/up'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([0])

f = glob.glob(r'40_data/left'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([1])
f = glob.glob(r'40_data/right'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([1])

f = glob.glob(r'40_data/CW'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([2])

f = glob.glob(r'40_data/CCW'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([3])

f = glob.glob(r'40_data/VLR'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([4])

f = glob.glob(r'40_data/VRL'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([5])


f = glob.glob(r'40_data/non'+'/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        sensor.append(t)
        label.extend([6])

sensor = np.array(sensor)
print ('sensor shape is :',sensor.shape)

feature_name = [ "x["+str(i)+"]" for i in range((sensor.shape[1]*sensor.shape[2]))]
sensor = np.reshape(sensor,(sensor.shape[0],sensor.shape[1]*sensor.shape[2]))
label  = np.array(label)
print ('sensor shape after is :',sensor.shape)
print ('label shape is :',label.shape)
print ('label is :',label)
#%%

#%%-------- My functions
def errorRate(predict_y, real_y):
    err=0
    for i in range(real_y.shape[0]):
        if predict_y[i] != real_y[i]:
            err+=1
    error = err/real_y.shape[0]*100
    return error

def train_val_split(sensors, labels, valData_percent=0.2):
    labels = labels.reshape(len(labels),1)
    trainData_percent = 1-valData_percent
    dat = np.concatenate((sensors, labels), axis=1)
    trainData_size = int(np.round(dat.shape[0] * trainData_percent))
    np.random.shuffle(dat)
    train, val = dat[:trainData_size,:], dat[trainData_size:,:]
    train_x, train_y = train[:,:train.shape[1]-1], train[:,-1]
    val_x, val_y = val[:,:val.shape[1]-1], val[:,-1]
    return train_x, val_x, train_y.astype(int).flatten(), val_y.astype(int).flatten()

def classCount(labels):
    dict = {}
    for key in labels:
        dict[key] = dict.get(key, 0) + 1
    return dict     

def oneSd_gini(labels):
    counts = classCount(labels)
    gini = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(labels))
        gini += prob_of_lbl**2
    impurity = 1-gini
    return impurity

def oneSd_entropy(labels):
    counts = classCount(labels)
    entropy = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(labels))
        entropy += prob_of_lbl * np.log2(prob_of_lbl)
    entropy = entropy * -1
    return entropy   

def entropy_calculation(labels_right, labels_left): 
    I_Dp = oneSd_entropy(np.hstack([labels_right,labels_left]))
    total_data_length = len(labels_right) + len(labels_left)
    entropy_r = oneSd_entropy(labels_right)
    entropy_l = oneSd_entropy(labels_left)
    total_entropy = I_Dp - len(labels_right)/total_data_length * entropy_r - len(labels_left)/total_data_length *entropy_l 
    return total_entropy

def gini_calculation(labels_right, labels_left):
    I_Dp = oneSd_gini(np.hstack([labels_right,labels_left]))
    total_data_length = len(labels_right) + len(labels_left)
    gini_r = oneSd_gini(labels_right)
    gini_l = oneSd_gini(labels_left)
    total_gini = I_Dp - len(labels_right)/total_data_length * gini_r - len(labels_left)/total_data_length *gini_l 
    return total_gini
    
def threshold_finder(train_X,train_y,f,method):
    entropies=[]
    thresholds_candid = train_X[:,f]
    
    if method == 'gini':
        for threshold in thresholds_candid:
            left_cluster = train_y[np.where(thresholds_candid<=threshold)[0]]
            right_cluster = train_y[np.where(thresholds_candid>threshold)[0]]
            entropy = entropy_calculation(right_cluster,left_cluster)
            entropies.append(entropy)
        entropies = np.array(entropies)
        ans = thresholds_candid[np.argmax(entropies)]
        
    elif method == 'entropy':
        for threshold in thresholds_candid:
            left_cluster = train_y[np.where(thresholds_candid<=threshold)[0]]
            right_cluster = train_y[np.where(thresholds_candid>threshold)[0]]
            entropy = entropy_calculation(right_cluster,left_cluster)
            entropies.append(entropy)
        entropies = np.array(entropies)
        ans = thresholds_candid[np.argmax(entropies)]
    
    return ans, np.argmax(entropies)

def threshold_finder_from_many_features(train_X, train_y, mtd):
    maxs=[]
    anss=[]
    for i in range(train_X.shape[1]):
        ans, max_entropy = threshold_finder(train_X,train_y,i,method=mtd)
        anss.append(ans)
        maxs.append(max_entropy)
    bestFeature = np.argmax(np.array(maxs))
    threshold = anss[bestFeature]
    return bestFeature, threshold 

def separate_2(train_X, train_y, feature, threshold):
    thresholds_candid = train_X[:,feature]
    left_cluster_y = train_y[np.where(thresholds_candid<=threshold)[0]]
    left_cluster_x = train_X[np.where(thresholds_candid<=threshold)[0]]
    
    right_cluster_x = train_X[np.where(thresholds_candid>threshold)[0]]
    right_cluster_y = train_y[np.where(thresholds_candid>threshold)[0]]
    return left_cluster_x, left_cluster_y, right_cluster_x, right_cluster_y
#%%

#%%-------- testing data preprocess
test_x = []
f = glob.glob(r'testdata/*.csv')
for i in range(len(f)):
    t = read_csv(f[i])
    if (len(t[0])) == 40:
        t = feature_ex(t)
        test_x.append(t)
test_x = np.array(test_x)
test_x = np.reshape(test_x,(test_x.shape[0],test_x.shape[1]*test_x.shape[2]))
#%%

#%%-------- training and validation data split
train_X, val_X, train_y, val_y = train_val_split(sensor, label, valData_percent=0.2)
print ('train_y is :',train_y)
print ('val_y is :',val_y)
#%%

#%%-------- build decision tree and multi-trees
class decisionTree:
    def __init__(self, max_depth, method='entropy'):
        self.depth = 0
        self.max_depth = max_depth
        self.method = method
        
    def fit(self, x, y, par_node = {}, depth=0):
        if par_node is None:   
            print('par_node is None')
            return None
        elif len(y) == 0:   
            print('len(y) is 0')
            return None
        elif self.all_same(y):  
            print('Same gestures, no need to classiy further')
            return {'Who':y[0]}
        elif depth >= self.max_depth:   
            print('Over max depth, still remains different gestures')
            return {'Who':np.argmax(np.bincount(y))}
        
        else:
            f, t_f = threshold_finder_from_many_features(x, y, mtd=self.method)
            left_x, left_y, right_x, right_y = separate_2(x, y,  f, t_f)
            
            par_node = {'feature': f, 'Threshold':t_f, 'Who': np.argmax(np.bincount(y))}
            par_node['left'] = self.fit(left_x, left_y, {}, depth+1)   
            par_node['right'] = self.fit(right_x, right_y, {}, depth+1)  
        
            self.depth += 1
            self.trees = par_node  
            return par_node  
        
    def all_same(self, items):
        return all(x == items[0] for x in items)
    
    def predict(self, val_X):
        anss=[]
        for i in range(val_X.shape[0]):
            row = val_X[i,:]
            ans = self._predict(self.trees, row)
            anss.append(ans)
        anss = np.array(anss)
        self.MyPrediction = anss
        return anss
    
    def _predict(self,result,row):
        while result.get('Threshold'):    
            if row[result['feature']] <= result['Threshold']:
                result = result['left']
                # print('L')
            elif row[result['feature']] > result['Threshold']:
                result = result['right']
                # print('R')   
        return result['Who']
    
    def score(self, val_y):
        err=0
        for i in range(val_y.shape[0]):
            if self.MyPrediction[i] != val_y[i]:
                err+=1
        error = err/val_y.shape[0]*100
        return 100-error
    
class multiDecisionTrees:
    # tree_num: how many trees you want
    # max_septh: maximum depth of trees
    # method: 'entropy' or 'gini'
    def __init__(self, tree_num, max_depth, method='entropy'):
        self.max_depth = max_depth
        self.tree_num = tree_num
        self.method = method
        self.tree_generator()
        self.myPrediction = None
        
    def tree_generator(self):
        trees = []
        for i in range(self.tree_num):
            trees.append(decisionTree(max_depth=self.max_depth, method=self.method))     
        self.myTrees = trees
        
    def trees_fit(self,x,y,sample_rate):
        # sample_rate: how many training data in each tree.
        # We need diverse training set
        for tree in self.myTrees:
            train_x, val_x, train_y, val_y = train_val_split(x,y,valData_percent=1-sample_rate)
            tree.fit(train_x,train_y)
    
    def trees_predict(self,x):
        predictions = []
        for tree in self.myTrees:
            prediction = tree.predict(x)
            predictions.append(prediction)
            
        predictions = np.array(predictions)
        ans = []
        for i in range(predictions.shape[1]):
              final_pred = np.argmax(np.bincount(predictions[:,i]))
              ans.append(final_pred)
        ans = np.array(ans) 
        self.myPrediction = ans
        return self.myPrediction

    def score(self,y):
        err=0
        for i in range(y.shape[0]):
            if self.myPrediction[i] != y[i]:
                err+=1
        error = err/y.shape[0]*100
        return 100-error  

# single tree
clf = decisionTree(max_depth=13, method='gini')
clf.fit(train_X, train_y)

# multiple trees
# clf_forest = multiDecisionTrees(tree_num=7, max_depth=13, method='gini')
# clf_forest.trees_fit(train_X, train_y, sample_rate=0.8)


#%%

#%%-------- predict training data or testing data
# single tree
val_y_predicted = clf.predict(val_X)
train_X_predicted = clf.predict(train_X)
test_X_predicted = clf.predict(test_x)

# multiple trees
# val_y_predicted_trees = clf_forest.trees_predict(val_X)
# train_X_predicted_trees = clf_forest.trees_predict(train_X)

#%%

#%%-------- accuracy
# single tree
acc = accF(val_y_predicted,val_y)
print ('val_y acc is :',acc)
acc = accF(train_X_predicted,train_y)
print ('train_y acc is :',acc)
print('test_x_predicted (contest on Kaggle):',test_X_predicted)

# multiple trees
# acc = accF(val_y_predicted_trees,val_y)
# print ('val_y acc is :',acc)
# acc = accF(train_X_predicted_trees,train_y)
# print ('train_y acc is :',acc)

#%%

# %%--------training model --> randomForest
# forest = multiDecisionTrees(tree_num=7, max_depth=13, method='entropy')
# forest.trees_training(train_X_ori, train_y_ori, sample_rate=0.8)
# pred_trees = forest.trees_predict(test_x)


#%% scoring
# highest_score = np.array([3,0,2,5,4,4,4,5,6,6,2,3,1,1,4,0,1,5,5,6,4,4,5,6,0,0,1,1,6,2,1,0,0,1,6,3,3,3,
# 4,4,6,4,6,4,5,5,4,4,3,6,3,2,0,0,5,4,5,4,1,1,0,5,0,5,0,2,2,3,0,4,5])

# err = errorRate(highest_score, pred_trees)

#%%
# test_depth = np.array([5,7,9,11,13,15,17,19,21])
# final_score=[]
# for depth in test_depth:
#     scores=[]
#     train_X_ori, val_X_ori, train_y_ori, val_y_ori = train_val_split(sensor, label, valData_percent=0.2)
#     for i in range(5):
#         clf_1 = decisionTree(max_depth=depth)
#         result_1 = clf_1.fit(train_X_ori,train_y_ori)
#         MyPrediction_1 = clf_1.predict(val_X_ori)
#         score = clf_1.score(val_y_ori)
#         scores.append(score)
#     scores_mean = np.array(scores).mean()
#     final_score.append(scores_mean)
    
# plt.plot(test_depth, final_score)

