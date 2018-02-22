import pandas as pd
import copy

# RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
# Geography Col. 1 -> HotLabel
# Gender Col. 2 -> LabelEncoder
dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values.reshape(-1, 1)

# TODO: Dados Categóricos ....
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

geographySet = set()
geographyOriginal = copy.copy(x[:, 1])
geographyEncoder = LabelEncoder()
x[:, 1] = geographyEncoder.fit_transform(x[:, 1])
for s, v in zip(geographyOriginal, x[:, 1]):
    geographySet.add("{} {}".format(s, v))
print(geographySet)

sexoSet = set()
sexoOriginal = copy.copy(x[:, 2])
sexoEncoder = LabelEncoder()
x[:, 2] = sexoEncoder.fit_transform(x[:, 2])
for s, v in zip(sexoOriginal, x[:, 1]):
    sexoSet.add("{} {}".format(s, v))
print(sexoSet)
print(x)

# Non-categorical features are always stacked to the right of the matrix., ou seja, as novas colunas aparecem na esquerda
oneHotEncoder = OneHotEncoder(categorical_features=[1])
x = oneHotEncoder.fit_transform(x).toarray()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Normalização
from sklearn.preprocessing import StandardScaler, LabelEncoder

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)
# y_test = sc_y.transform(y_test)
