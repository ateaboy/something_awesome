import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split

#import the civil data

df = pd.read_excel("civil_unrest.xlsx")

civil_X = []

#convert catagorical data to dummy variables  
df_country = pd.get_dummies(df['GP3'])
del df['GP3']
df = pd.concat([df, df_country], axis = 1)

#now convert input data into 2D array
for index, row in df.iterrows():
    row_obj = []
    
    for name in df.columns:
        if (str(name) != "N_INJURD"):
            row_obj.append(row[str(name)])
    
    civil_X.append(row_obj)

# convert output data into 1D array
civil_Y = []
for index, row in df.iterrows():
    civil_Y.append(row['N_INJURD'])

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(civil_X, civil_Y, test_size=0.2)

# Create linear regression object
clf = linear_model.LinearRegression()

# Train the model using the training sets
clf.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = clf.predict(X_test)

r2_score = clf.score(X_test, y_test)
print(r2_score*100,'%')


