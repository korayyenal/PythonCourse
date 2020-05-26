import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn import svm
from sklearn import metrics

# Read data with genres
songs = pd.read_csv('/Users/korayyenal/Downloads/echonestdataset.csv')

# Read metrics
metrics = pd.read_json('/Users/korayyenal/Downloads/metrics.json', precise_float=True)

# Merge relevant columns
merged_songs = pd.merge(metrics, songs[['track_id', 'genre_top']], on='track_id')

#Correlation Matrix
corr_metrics = merged_songs.corr()
corr_metrics.style.background_gradient()

# Define our features 
song_feats = merged_songs.drop(['track_id', 'genre_top'], axis=1)
# Define genre labels
labels = merged_songs['genre_top']

# Scale features and set values to a new variable
scaler = StandardScaler()
scaled_train_feats = scaler.fit_transform(song_feats)

%matplotlib inline

pca = PCA()
pca.fit(scaled_train_feats)
exp_variance = pca.explained_variance_ratio_

fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('Principal Component number')

#cumulative explained variance
cum_exp_variance = np.cumsum(exp_variance)

#Plot with a dashed line at 0.90.
fig, ax = plt.subplots()
ax.plot(cum_exp_variance)
ax.axhline(y=0.9, linestyle='--')
n_components = 6

#PCA 
pca = PCA(n_components, random_state=10)
pca.fit(scaled_train_feats)
pca_projection = pca.transform(scaled_train_feats)

# Split data
train_feats, test_feats, train_labels, test_labels = train_test_split(pca_projection, 
                                                                            labels,
                                                                        random_state=10)

# Train decision tree
dt = DecisionTreeClassifier(random_state=10)
dt.fit(train_feats, train_labels)

# Predict labels for test 
predict_dt = dt.predict(test_feats)
dt.score(test_feats, test_labels)

#logistic regression and predict labels for test
lr = LogisticRegression(random_state=10)
lr.fit(train_feats, train_labels)
predict_lr = lr.predict(test_feats)

#Classification Report
classification_dt = classification_report(test_labels, predict_dt)
classification_lr = classification_report(test_labels, predict_lr)

print("Decision Tree: \n", classification_dt)
print("Logistic Regression: \n", classification_lr)

# Subset hip-hop & rock songs separately
hiphopsongs = merged_songs[merged_songs['genre_top']=='Hip-Hop']
rocksongs = merged_songs[merged_songs['genre_top']=='Rock']

# sample equal number of rocks songs as hip-hop songs
rocksongs = rocksongs.sample(len(hiphopsongs), random_state=10)
hiphop_rock_merged = pd.concat([rocksongs, hiphopsongs])

# creating a balanced dataframe
song_feats = hiphop_rock_merged.drop(['genre_top', 'track_id'], axis=1) 
labels = hiphop_rock_merged['genre_top']
pca_projection = pca.fit_transform(scaler.fit_transform(song_feats))

# Redefine train and test with the balanced data
train_feats, test_feats, train_labels, test_labels = train_test_split(pca_projection, labels, random_state=10)

# Train Decision Tree on the balanced 
dt = DecisionTreeClassifier(random_state=10)
dt.fit(train_feats, train_labels)
predict_dt = dt.predict(test_feats)

# Train LR on the balanced 
lr = LogisticRegression(random_state=10)
lr.fit(train_feats, train_labels)
predict_lr = lr.predict(test_feats)

# Compare two models
print("Decision Tree: \n", classification_report(test_labels, predict_dt))
print("Logistic Regression: \n", classification_report(test_labels, predict_lr))

#k-fold cv
kf = KFold(10,shuffle=True, random_state=10)

dt = DecisionTreeClassifier(random_state=10)
lr = LogisticRegression(random_state=10)

# Train model by k-fold cv
dt_score = cross_val_score(dt, pca_projection, labels, cv=kf)
lr_score = cross_val_score(lr, pca_projection, labels, cv=kf)

# Print scores
print("Decision Tree:", dt_score, "\nLogistic Regression:", lr_score)

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

train_feats, test_feats, train_labels, test_labels = train_test_split(pca_projection, labels, random_state=10)

#Train the model using the training sets
clf.fit(train_feats, train_labels)

predict_svm = clf.predict(test_feats)

classification_svm = classification_report(test_labels, predict_svm)
print("Support Vector: \n", classification_report(test_labels, predict_svm))
