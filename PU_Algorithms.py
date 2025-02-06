import pandas as pd
import numpy as np

# Keep the original targets safe for later
y_orig = y.copy()
# Unlabel a certain number of data points
hidden_size = 1000 ### ENTER A NUMBER HERE ###
y.loc[
    np.random.choice(
        y[y == 1].index, 
        replace = False, 
        size = hidden_size
    )
] = 0

################################################################
# 1. 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators = 1000,  # 1000 trees
    n_jobs = -1           # Use all CPU cores
)
rf.fit(X, y)

results = pd.DataFrame({
    'truth'      : y_orig,                    # True labels
    'label'      : y,                         # Labels shown to models
    'output_std' : rf.predict_proba(X)[:,1]   # Random forest's scores
}, columns = ['truth', 'label', 'output_std'])

# 2. PU bagging
from sklearn.tree import DecisionTreeClassifier
n_estimators = 1000
estimator = DecisionTreeClassifier()

iP = y[y > 0].index
iU = y[y <= 0].index

num_oob = pd.DataFrame(np.zeros(shape = y.shape), index = y.index)
sum_oob = pd.DataFrame(np.zeros(shape = y.shape), index = y.index)

for _ in range(n_estimators):
    # Get a bootstrap sample of unlabeled points for this round
    ib = np.random.choice(iU, replace = True, size = len(iP))

    # Find the OOB data points for this round
    i_oob = list(set(iU) - set(ib))

    # Get the training data (ALL positives and the bootstrap
    # sample of unlabeled points) and build the tree
    Xb = X[y > 0].append(X.loc[ib])
    yb = y[y > 0].append(y.loc[ib])
    estimator.fit(Xb, yb)

    # Record the OOB scores from this round
    sum_oob.loc[i_oob, 0] += estimator.predict_proba(X.loc[i_oob])[:,1]
    num_oob.loc[i_oob, 0] += 1
    
results['output_bag'] = sum_oob / num_oob

# 3. A custom PU classifier
from baggingPU import BaggingClassifierPU
bc = BaggingClassifierPU(
    DecisionTreeClassifier(), n_estimators = 1000, n_jobs = -1, 
    max_samples = sum(y)  # Each training sample will be balanced
)
bc.fit(X, y)
results['output_bag'] = bc.oob_decision_function_[:,1]

# 4. A two-step approach
ys = 2 * y - 1
pred = rf.predict_proba(X)[:,1]
range_P = [min(pred * (ys > 0)), max(pred * (ys > 0))]
# step 1
iP_new = ys[(ys < 0) & (pred >= range_P[1])].index
iN_new = ys[(ys < 0) & (pred <= range_P[0])].index
ys.loc[iP_new] = 1
ys.loc[iN_new] = 0
# step 2
rf2 = RandomForestClassifier(n_estimators = 1000, n_jobs = -1)

for i in range(10):
    # If step 1 didn't find new labels, we're done
    if len(iP_new) + len(iN_new) == 0 and i > 0:
        break
    print(
        'Step 1 labeled %d new positives and %d new negatives.'
         % (len(iP_new), len(iN_new))
    )
    print('Doing step 2... ', end = '')

    # Retrain on new labels and get new scores
    rf2.fit(X, ys)
    pred = rf2.predict_proba(X)[:,-1]

    # Find the range of scores given to positive data points
    range_P = [min(pred * (ys > 0)), max(pred * (ys > 0))]

    # Repeat step 1
    iP_new = ys[(ys < 0) & (pred >= range_P[1])].index
    iN_new = ys[(ys < 0) & (pred <= range_P[0])].index
    ys.loc[iP_new] = 1
    ys.loc[iN_new] = 0

results['output_stp'] = pred

# 5. Averaging
results['output_all'] = results[[
    'output_std', 'output_bag', 'output_stp'
]].mean(axis = 1)

############################ Random Forest #################
#Experiments with artificial data
#Circles
import pandas as pd
import numpy as np

from sklearn.datasets import make_circles
X, y = make_circles(
    n_samples = 6000, noise = 0.1, 
    shuffle = True, factor = .65
)
X = pd.DataFrame(X, columns = ['feature1', 'feature2'])
y = pd.Series(y)

X.to_csv('X.csv')
y.to_csv('y.csv')
y_orig = y.copy()
hidden_size = 1000

y.loc[
    np.random.choice(
        y[y == 1].index, 
        replace = False, 
        size = hidden_size
    )
] = 0

y.sum(axis=0)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators = 1000,  # 1000 trees
    n_jobs = -1           # Use all CPU cores
)
rf.fit(X, y)

ypre=rf.predict_proba(X)
ypre.sum(axis=0)
X.to_csv("X.csv")
y.to_csv("y.csv")
y_orig.to_csv("y_orig.csv")
np.savetxt("ypre.csv", ypre, delimiter=",")
#yprepd=pd.DataFrame(data=ypre[:,:])
#yprepd.to_csv("yprepd.csv")

################################## Real mutation data matrix from mutstrvarSXY.txt at /genetics/tenX/databam
import matplotlib
matplotlib.use('agg')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

mutstrvarXY=pd.read_csv('mutstrvarSXY.txt', sep="\t")
mutstrvarXY.dtypes
#mutstrvarXY=mutstrvarXY.sample(frac=1)
mutstrvarXY=mutstrvarXY.sample(frac=1).reset_index(drop=True)
m,n=mutstrvarXY.shape
m,n  #mutstrvarXY=mutstrvarXY.iloc[0:m-4,4:]
mutstrvarXY=mutstrvarXY.iloc[:,5:]
mutstrvarXY.dtypes
m,n=mutstrvarXY.shape
m,n
#mutstrvarXY=mutstrvarXY.drop(mutstrvarXY.columns[[n-1]],axis=1)
#mutstrvarXY=mutstrvarXY.rename(columns={'mutstrvar': 'LABEL'})
#mutstrvarXY['LABEL']=mutstrvarXY['LABEL'].map({0:1,1:0})

# mutstrvarXY=mutstrvarXY.values
# np.random.shuffle(mutstrvarXY)

mutstrvarX=mutstrvarXY.iloc[:,0:n-2]
mutstrvarY=mutstrvarXY.iloc[:,n-2] #mutstrvarY=mutstrvarXY.iloc[:,n-1]
mutstrvarY.sum()
#mutstrvarY=mutstrvarY.map({0:1,1:0})

###standarization/normalization dataframe X
m,n=mutstrvarX.shape
colsum_mean=mutstrvarX.mean(axis=0)
col_std=mutstrvarX.std(axis=0)
mutstrvarX_std=mutstrvarX
mutstrvarX_std=(mutstrvarX-colsum_mean)/col_std
mutstrvarX_std.sum(axis=0)
mutstrvarX_nor=mutstrvarX
mutstrvarX_nor=(mutstrvarX-colsum_mean)/(mutstrvarX.max()-mutstrvarX.min())
mutstrvarX_nor.sum(axis=0)

# mutstrvarY = mutstrvarY.squeeze()
# mutstrvarY = mutstrvarY.astype(int)
# mutstrvarY.sum(axis=0)

#### check the value of each commpond of matrix is >-20 and <200
#np.issubdtype(mutstrvarX_std.iloc[0][0], np.float64)

#for i in range(m):
#	for j in range(n):
#		mutstrvarX.iloc[i][j]=mutstrvarX.iloc[i][j]-colmean[j]
#	print(mutstrvarX.iloc[i])


RF = RandomForestClassifier(
    n_estimators = 1000,  # 1000 trees
    #max_features=20,
    #max_depth=5,
    #min_samples_leaf=2,
    random_state=42,
    #n_jobs = -1           # Use all CPU cores
)


#ET = ExtraTreesClassifier(
#    n_estimators = 1000,  # 1000 trees
#    n_jobs = -1           # Use all CPU cores
#)

###skip the train, test processing
#from sklearn.model_selection import train_test_split
#m,n=mutstrvarXY.shape
#mutstrvarY=mutstrvarXY.iloc[:,n-1]
#mutstrvarY=mutstrvarY.map({0:1,1:0})
#mutstrvarY.sum()
#train_features, test_features, train_labels, test_labels = train_test_split(mutstrvarX, mutstrvarY, test_size = 0.25, random_state = 42)
#print('Training Features Shape:', train_features.shape)
#print('Training Labels Shape:', train_labels.shape)
#print('Testing Features Shape:', test_features.shape)
#print('Testing Labels Shape:', test_labels.shape)
### end the skip

# New random forest with only the common important features
rf_most_important = RandomForestClassifier(n_estimators= 1000, random_state=42)
feature_list = list(mutstrvarX.columns)
# Extract the common most important features
#important_indices = [feature_list.index('DP_N'),feature_list.index('DP_T'),feature_list.index('GC'),
#		feature_list.index('HRun'), feature_list.index('LowMQ0_N'), feature_list.index('LowMQ0_T'), feature_list.index('LowMQ10_N'),
#		feature_list.index('LowMQ10_T'), feature_list.index('MQ_N'), feature_list.index('MQ_T'), feature_list.index('MQ0_N'),
#		feature_list.index('MQ0_T'),feature_list.index('NCC'),feature_list.index('SOR_N'),feature_list.index('SOR_T'), 
#		feature_list.index('SPV'),feature_list.index('SSC'),feature_list.index('Var_SNP'),feature_list.index('Var_indel'),
#		feature_list.index('Var_num_N'),feature_list.index('Var_num_T'),feature_list.index('fisherETp_NT')]
important_indices = ['DP_N','DP_T','GC','HRun','LowMQ0_N','LowMQ0_T','LowMQ10_N','LowMQ10_T','MQ_N','MQ_T','MQ0_N','MQ0_T',
                     'NCC','SOR_N','SOR_T','SPV','SSC','Var_SNP','Var_indel','Var_num_N','Var_num_T','fisherETp_NT']
train_important = mutstrvarX[important_indices]
train_labels=mutstrvarY
#test_important = test_features[important_indices]
#train_important=train_features[['NCC','AC','SomaticEVS','TLOD','str','NT','SGT','NLOD','ECNT','HCNT','SSC','mut ','QSS_NT','fisherETp_NT',
#	'DP_N','DP_T','GC','GPV','MQ_N','MQ_T','QSI','QSI_NT','QSS','RU','SOR_N','SPV','TQSS','TQSS_NT','Var_num_T','Var_rate_T','var']] 

rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf.fit(train_important, train_labels)
predictions = rf.predict(train_important)
predictions_proba = rf.predict_proba(train_important)
np.savetxt("train_predic.csv", predictions_proba, delimiter=",")
np.savetxt("train_labels.csv", train_labels, delimiter=",")

#test_labels=test_labels.drop(test_labels.columns[[0]],axis=1)
#test_labels=test_labels.reset_index(drop=True)
bothone=0
for i in range(len(predictions)):
	if predictions[i]==1 and train_labels[i]==1:
		bothone=bothone+1

print("number of prediction==train_labels==1:", bothone)
print("Positive Prediction=:", bothone/predictions.sum())

bothone=0
m,n=mutstrvarXY.shape
train_GroundT=mutstrvarXY.iloc[:,n-1]
for i in range(len(predictions)):
	if predictions[i]==1 and train_GroundT[i]==1:
		bothone=bothone+1

print("number of prediction==train_GroundT==1:", bothone)
print("Sensitivity=:", bothone/train_GroundT.sum())

#####display the tree
from sklearn.tree import export_graphviz
import pydot
tree = rf.estimators_[5]
feature_list = list(train_important.columns)
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

def accuracyPrediction(predictions, i, train_labels, train_GroundT):
	print("Tree Level:", i )
	prelabel=0; preGT=0
	for j in range(len(predictions)):
		if predictions[j]==1 and train_labels[j]==1:
			prelabel=prelabel+1
		if predictions[j]==1 and train_GroundT[j]==1:
			preGT=preGT+1
	print("number of prediction==train_labels==1:", prelabel)
	print("Label Pred rate=", prelabel/predictions.sum() )
	print("number of prediction==train_GroundT==1:", preGT)
	print("number of GroundTruth=", train_GroundT.sum())
	print("number of Pos Prediction=", predictions.sum() )
	sens=preGT/train_GroundT.sum()
	ppre=preGT/predictions.sum()
	print("Sensitivity=", sens)
	print("Positive Predictive=", ppre)
	F1=2*sens*ppre/(sens+ppre)
	print("F1=", F1)
	return F1


def predictionByProba(predictions_proba, num, probthr_l, probthr_h):
	predictions=predictions_proba[:,1].copy()
	thr_l=probthr_l #0
	thr_h=probthr_h #0.5
	for i in range(len(predictions_proba)):
		if predictions[i]>(thr_h+thr_l)/2: predictions[i]=1
		else: predictions[i]=0
	num_h=predictions.sum()
	if num>num_h*(1.05): thr_h=(thr_h+thr_l)/2  #if num>num_h*(1.05): thr_h=(thr_h+thr_l)/2
	elif num<num_h*(0.95): thr_l=(thr_h+thr_l)/2 #elif num<num_h*(0.95): thr_l=(thr_h+thr_l)/2 
	else: thr_h=thr_l=(thr_h+thr_l)/2
	if thr_h!=thr_l:
		predictions, thr_h=adjust_predictions(predictions_proba, num, thr_l, thr_h)
	else: print("number of pos:", predictions.sum())
	return predictions, thr_h
	

# find out the best depth of tree
# deep level 7 with (0.8996, 0.9206) or 8 with (0.9236, 0.9118)
maxF1=0
bestDepth=3
F1=0
minSampleSplit=2
for i in range(3, 22):
	for j in range(2, 20):
		rf_small = ExtraTreesClassifier(n_estimators=10, max_depth=i, min_samples_split=j, min_samples_leaf=2, min_impurity_split=1e-7  )
		#rf_small = RandomForestClassifier(n_estimators=10, max_depth=i, min_samples_leaf=2 )
		rf_small.fit(train_important, train_labels) #train_features
		#predictions = rf_small.predict(train_important)
		#print("prediction")
		#accuracyPrediction(predictions, i, train_labels, train_GroundT)
		predictions_proba = rf_small.predict_proba(train_important)
		predictions,thr=predictionByProba(predictions_proba, train_GroundT.sum(), 0, 0.49)
		print("prediction_proba with thr=", thr)
		F1=accuracyPrediction(predictions, i, train_labels, train_GroundT)
		if F1>maxF1:
			maxF1=F1
			bestDepth=i
			minSampleSplit=j
			
print("max F1=%s, depth=%s, min_sample_split=%s" % (maxF1, bestDepth, minSampleSplit) )	
rf_small = ExtraTreesClassifier(n_estimators=1000, max_depth=bestDepth, min_samples_split=minSamplesSplit, min_samples_leaf=2 )
rf_small.fit(train_important, train_labels) #train_features
predictions_proba = rf_small.predict_proba(train_important)
predictions,thr=predictionByProba(predictions_proba, train_GroundT.sum(), 0, 0.49)
print("prediction_proba with thr=", thr)
F1=accuracyPrediction(predictions, bestDepth, train_labels, train_GroundT)

######## save fitted model into file
rf_small.set_params(n_jobs=1)

import os
model_dir='trained_model'
os.mkdir(model_dir)
from sklearn.externals import joblib
joblib.dump(rf_small, model_dir + '/pu_rf')
modelrf = joblib.load(model_dir + '/pu_rf')

mutstrvarYrf=rf_small.predict_proba(train_important) #mutstrvarYmodelrf=modelRF.predict_proba(mutstrvarX)
predictions,thr=predictionByProba(mutstrvarYrf, train_GroundT.sum(), 0, 0.49)
predictions.sum()
np.savetxt("mutstrvarYrf.csv",  mutstrvarYrf, delimiter=",")
mutstrvarXY.to_csv("mutstrvarXY.csv")

# Get numerical feature importances
importances = list(rf_small.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


Whole features			Common features	
Variable	Importance	Variable	Importance
NCC		0.14		NCC		0.71
AC		0.12		SSC		0.09
SomaticEVS	0.09		SPV		0.04
TLOD		0.08		LowMQ0_N	0.02
str		0.06		SOR_N		0.02
NT		0.05		var_indel	0.02
SGT		0.05		LowMQ0_T	0.01
NLOD		0.04		LowMQ10_N	0.01
ECNT		0.03		LowMQ10_T	0.01
HCNT		0.03		MQ_N		0.01
SSC		0.03		MQ0_N		0.01
mut		0.03		Var_SNP		0.01
QSS_NT		0.02		fisherETp_NT	0.01
fisherETp_NT	0.02		DP_N		0
DP_N		0.01		DP_T		0
DP_T		0.01		GC		0
GC		0.01		HRun		0
GPV		0.01		MQ_T		0
MQ_N		0.01		MQ0_T		0
MQ_T		0.01		SOR_T		0
QSI		0.01		Var_num_N	0
QSI_NT		0.01		Var_num_T	0
QSS		0.01			
RU		0.01			
SOR_N		0.01			
SPV		0.01			
TQSS		0.01			
TQSS_NT		0.01			
Var_num_T	0.01			
Var_rate_T	0.01			
var		0.01			
AF		0			
FS_N		0			
FS_T		0			
HRun		0			
IC		0			
IHP		0			
LowMQ0_N	0			
LowMQ0_T	0			
LowMQ10_N	0			
LowMQ10_T	0			
MAX_ED		0			
MQ0_N		0			
MQ0_T		0			
PercentNBase_N	0			
PercentNBase_T	0			
RC		0			
RPA		0			
ReadPosRankSum	0			
SNVSB		0			
SOMATIC		0			
SOR_T		0			
STR		0			
TQSI		0			
TQSI_NT		0			
Var_SNP		0			
var_indel	0			
Var_num_N	0			
Var_rate_N	0			
############################################################################### end real data analysis


########################################################################## display dicision tree
# Pandas is used for data manipulation
import matplotlib
matplotlib.use('agg')
import os
cwd = os.getcwd()

import pandas as pd
# Read in data and display first 5 rows
features = pd.read_csv('temps.csv')
features.head(5)
print('The shape of our features is:', features.shape)   #(348,12)
features=features.drop(features.columns[[8,9,10]], axis=1)
features.head(5)
print('The shape of our features is:', features.shape)   #(348,9)
# Descriptive statistics for each column
features.describe()
# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(features)
# Display the first 5 rows of the last 12 columns
features.iloc[:,5:].head(5)

# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(features['actual'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('average')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))
#Average baseline error:  5.06 degrees.

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
#Mean Absolute Error: 3.83 degrees.

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
#Accuracy: 93.99 %.

# Import tools needed for visualization
from sklearn.tree import export_graphviz

import graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz

#pip install graphviz

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

#exit()
#dot tree.dot -Tpng -o tree.png

# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(train_features, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png')

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
important_indices = [feature_list.index('temp_1'), feature_list.index('average')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')
# Mean Absolute Error: 3.9 degrees.
# Accuracy: 93.8 %.

# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt
%matplotlib inline
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
plt.show()
plt.savefig('VariableImportance.png', bbox_inches = "tight")
plt.close()

# Use datetime for creating date objects for plotting
import datetime
# Dates of training values
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]
# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
# Dates of predictions
months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]
# Column of dates
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})
# Plot the actual values
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()
# Graph labels
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');
plt.show()
plt.savefig('ActualVsPredictedValues.png', bbox_inches = "tight")
plt.close()

# Make the data accessible for plotting
true_data['temp_1'] = features[:, feature_list.index('temp_1')]
true_data['average'] = features[:, feature_list.index('average')]
true_data['friend'] = features[:, feature_list.index('friend')]
# Plot all the data as lines
plt.plot(true_data['date'], true_data['actual'], 'b-', label  = 'actual', alpha = 1.0)
plt.plot(true_data['date'], true_data['temp_1'], 'y-', label  = 'temp_1', alpha = 1.0)
plt.plot(true_data['date'], true_data['average'], 'k-', label = 'average', alpha = 0.8)
plt.plot(true_data['date'], true_data['friend'], 'r-', label = 'friend', alpha = 0.3)
# Formatting plot
plt.legend(); plt.xticks(rotation = '60');
# Lables and title
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual Max Temp and Variables');
plt.show()
plt.savefig('MaxumumTemperature.png', bbox_inches = "tight")
plt.close()