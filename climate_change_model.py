
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split

#import the climate data as pandas dataset

df = pd.read_excel("climate_change.xlsx")

climate_X = []

#convert catagorical data to dummy variables 
df_country = pd.get_dummies(df['country'])
del df['country']
df = pd.concat([df, df_country], axis = 1)

#now convert input data into 2D array
for index, row in df.iterrows():
    row_obj = []
    
    for name in df.columns:
        #ignore output variable 
        if (str(name) != "average_temp"):
            row_obj.append(row[str(name)])
    
    climate_X.append(row_obj)

# convert output data into 1D array
climate_Y = []
for index, row in df.iterrows():
    climate_Y.append(row['average_temp'])

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(climate_X, climate_Y, test_size=0.2)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

#calculate r2 score
r2_score = regr.score(X_test, y_test)
print(r2_score*100,'%')



