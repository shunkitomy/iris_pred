from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
# from sklearn.externals import joblib
from sklearn.metrics import classification_report, accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

model = MLPClassifier(solver='sgd', random_state=0, max_iter=3000)

model.fit(X_train, y_train)
pred = model.predict(X_test)

# joblib.domp(model, 'nn.pkl', compress=True)

print('result:', model.score(X_test, y_test))
print(classification_report(y_test, pred))
