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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Merge relevant columns
merged_songs = pd.merge(metrics, songs[['track_id','Type']], on='track_id')
# Define our features 
song_feats = merged_songs.drop(['track_id','Type'], axis=1)
# Define genre labels
labels = merged_songs['Type']

hiphopsongs = merged_songs[merged_songs['Type']==0]
rocksongs = merged_songs[merged_songs['Type']==1]
#Sampling equal numbers from both
rocksongs = rocksongs.sample(len(hiphopsongs), random_state=10)
hiphop_rock_merged = pd.concat([rocksongs, hiphopsongs])

# Scale features and set values to a new variable
scaler = StandardScaler()
scaled_train_feats = scaler.fit_transform(song_feats)

n_components = 6

#PCA 
pca = PCA(n_components, random_state=10)
pca.fit(scaled_train_feats)
pca_projection = pca.transform(scaled_train_feats)
# creating a balanced dataframe
song_feats = hiphop_rock_merged.drop(['track_id','Type'], axis=1) 
labels = hiphop_rock_merged['Type']
pca_projection = pca.fit_transform(scaler.fit_transform(song_feats))

X_train, X_test, y_train, y_test = train_test_split(song_feats, labels, random_state=10)
# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(8, activation='relu', input_shape=(8,)))

# Add one hidden layer 
model.add(Dense(4, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))

# Model output shape
model.output_shape

# Model summary
model.summary()

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)

y_pred = model.predict(X_test)

score = model.evaluate(X_test, y_test,verbose=1)

print(score)
y_pred_2 = np.around(y_pred, decimals=0)
print(y_pred_2)

confusion_matrix(y_test, y_pred_2)

classification_nn = classification_report(y_test, y_pred_2)
print("Neural Nets: \n", classification_nn)
