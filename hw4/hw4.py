import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

tt = pd.read_csv('immSurvey.csv')
tt.head()
alphas = tt.stanMeansNewSysPooled
sample = tt.textToSend


vec = CountVectorizer()
X = vec.fit_transform(sample)

pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

#TF–IDF
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

Xtrain, Xtest, ytrain, ytest = train_test_split(X, alphas,
random_state=1)

rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=rbf, alpha=1e-8)

gpr.fit(Xtrain.toarray(), ytrain)

#posterior predictive mean and covariance
mu_s, cov_s = gpr.predict(Xtest.toarray(), return_cov=True)

#correlation between test and mus
model = np.corrcoef(ytest, mu_s)
print(np.corrcoef(ytest, mu_s))

#Bigram
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
#Convert to Numeric Data
X2 = bigram_vectorizer.fit_transform(sample)
X2train, X2test, y2train, y2test = train_test_split(X2, alphas,
                                                random_state=1)

rbf2 = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr2 = GaussianProcessRegressor(kernel=rbf, alpha=1e-8)

gpr2.fit(X2train.toarray(), y2train)

# posterior predictive mean and covariance
mu_s2, cov_s2 = gpr2.predict(X2test.toarray(), return_cov=True)

#correlation between test2 and mus2
model2=np.corrcoef(y2test, mu_s2)
print(np.corrcoef(y2test, mu_s2))

print ("With the second model, the correlation changed " +str(round(((model2 / model)[1,0]-1)*100,2)) +" % compared to the first model" )
