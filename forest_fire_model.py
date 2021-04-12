import matplotlib.pyplot as plt

from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split

#import the fire data

df = pd.read_excel("forest_fire.xlsx")

fire_X = []

#convert catagorical data to dummy variables  
df_month = pd.get_dummies(df['month'])
del df['month']
df = pd.concat([df, df_month], axis = 1)

#now convert input data into 2D array
for index, row in df.iterrows():
    row_obj = []
    
    for name in df.columns:
        if (str(name) != "area"):
            row_obj.append(row[str(name)])
    
    fire_X.append(row_obj)


# convert output data into 1D array
fire_Y = []
for index, row in df.iterrows():
    fire_Y.append(row['area'])

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(fire_X, fire_Y, test_size=0.2)

# Create linear regression object
regr = linear_model.Lasso()
# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

r2_score = regr.score(X_test, y_test)
print(r2_score*100,'%')
