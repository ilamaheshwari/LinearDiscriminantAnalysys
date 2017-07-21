import pandas as pd
dataset = pd.read_csv('/root/Downloads/banknotes.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,6].values

X=pd.DataFrame(X)

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
X[0]=label.fit_transform(X[0])


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
X_train=lda.fit_transform(X_train,y_train)
X_test=lda.transform(X_test)

from sklearn.linear_model import LogisticRegression
re=LogisticRegression(random_state=0)
re.fit(X_train,y_train)
ypred=re.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,ypred)