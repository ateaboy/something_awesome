import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split

#import the mental health data

df = pd.read_csv("mental_health_real.csv")

mental_X = []

#convert catagorical data to dummy variables  
df_country = pd.get_dummies(df['Location'])
df_gender = pd.get_dummies(df['Gender'])
del df['Location']
del df['Gender']
df = pd.concat([df, df_country, df_gender], axis = 1)

#now convert input data into 2D array
for index, row in df.iterrows():
    row_obj = []
    
    for name in df.columns:
        if (str(name) != "Sought Treatment"):
            row_obj.append(row[str(name)])
    
    mental_X.append(row_obj)


# convert output data into 1D array
mental_Y = []
for index, row in df.iterrows():
    mental_Y.append(row['Sought Treatment'])


#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(mental_X, mental_Y, test_size=0.2)

# Create linear regression object
clf = linear_model.LogisticRegression()

# Train the model using the training sets
clf.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = clf.predict(X_test)

print("Accuracy score:")
print(accuracy_score(y_test, y_pred))
