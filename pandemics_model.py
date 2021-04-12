import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split

#import the pandemics data

df = pd.read_excel("pandemics.xlsx")

pandemics_X = []

#convert catagorical data to dummy variables  
df_country = pd.get_dummies(df['Country'])
df_wealth = pd.get_dummies(df['Wealth_of_nation'])
del df['Country']
del df['Wealth_of_nation']
df = pd.concat([df, df_country, df_wealth], axis = 1)

#now convert input data into 2D array
for index, row in df.iterrows():
    row_obj = []
    
    for name in df.columns:
        if (str(name) != "Covid cases total"):
            row_obj.append(row[str(name)])
    
    pandemics_X.append(row_obj)


# convert output data into 1D array
pandemics_Y = []
for index, row in df.iterrows():
    pandemics_Y.append(row['Covid cases total'])

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(pandemics_X, pandemics_Y, test_size=0.2)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

r2_score = regr.score(X_test, y_test)
print(r2_score*100,'%')
