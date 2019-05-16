import pandas as pd

datos = pd.read_csv('cc1000_gm_Zeus.csv')

#print(datos)

tabla = datos.iloc[:,5:]

#print(tabla)


# Split Data into Training and Test Sets

from sklearn.model_selection import train_test_split
# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split( tabla, datos.iloc[:,1], test_size=1/7.0, random_state=0)

# Standardize the Data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training set only.

scaler.fit(train_img)

# Apply transform to both the training set and the test set.

train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)


# Import and Apply PCA


from sklearn.decomposition import PCA
# Make an instance of the Model

pca = PCA(.95)

pca.fit(train_img)


# Apply the mapping (transform) to both the training set and the test set.


train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

from sklearn.linear_model import LinearRegression

# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'

reg = LinearRegression().fit(train_img, train_lbl)

print(reg.score(train_img, train_lbl))

print(reg.predict(test_img[0].reshape(1,-1)))

#for i in  range(10):
#   print(i)
#   print(reg.predict(train_img[i].reshape(1,-1)))
#   print(train_lbl[i])

# Measuring Model Performance

print(reg.score(test_img, test_lbl))


