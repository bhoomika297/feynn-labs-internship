import numpy as np
import pandas as pd


df = pd.read_csv('fake_job_postings.csv')

df1 = pd.DataFrame(x.toarray(),columns=cv.get_feature_names_out())
df.drop(["text"],axis=1,inplace=True)
main_df = pd.concat([df1,df],axis=1)

Y = main_df.iloc[:,-1]
X = main_df.iloc[:,:-1]

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=100,criterion="entropy")
model = rfc.fit(X_train,y_train)