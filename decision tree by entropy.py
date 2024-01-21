import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

data = load_wine()
feature_names = data.feature_names  # Feature names
class_names = data.target_names  # Class names


X = data.data  
target = data.target  

data = pd.DataFrame(X)
target=pd.DataFrame(target)


x_train,x_test,y_train,y_test=train_test_split(data, target, test_size = 0.33, random_state = 42)


training_data = pd.concat([x_train,y_train],axis = 1)

testing_data = pd.concat([x_test,y_test],axis = 1)


train=classifier.fit(x_train, y_train)

Pred=classifier.predict(x_test)
print("prediction",Pred)


# Flatten array1 to ensure shape compatibility
array1_flattened = y_test.values.flatten()

# Compare arrays element-wise
comparison = array1_flattened == Pred

# Print the comparison result
print(comparison)

conf = confusion_matrix(y_test, Pred)

sns.heatmap(conf, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show

target_names = ['0', '1','2']
report = classification_report(y_test, Pred, target_names=target_names)
print ("Classification report")
print(report)
              
# Decrease the figure size

fig, ax = plt.subplots(figsize=(12, 12))

# Plot the decision tree with custom styling
tree.plot_tree(train, ax=ax, filled=True, rounded=True, feature_names=feature_names, class_names=class_names, fontsize=8)

# Customize the plot aesthetics
ax.set_title("Decision Tree - Wine Dataset (using entropy)", fontsize=20)
ax.set_xlabel("Features", fontsize=14)
ax.set_ylabel("Wine Class", fontsize=14)

# Adjust the arrow properties
for arrow in ax.get_xticklines():
    arrow.set_markersize(5)

# Adjust the spacing between subplots if needed
plt.subplots_adjust(wspace=0.5, hspace=0.5)
# Display the plot
plt.show()

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0,max_depth = 3)

train=classifier.fit(x_train, y_train)

Pred=classifier.predict(x_test)
conf = confusion_matrix(y_test, Pred)
conf

sns.heatmap(conf, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (depth 3)')
plt.show

target_names = ['0', '1','2']
report = classification_report(y_test, Pred, target_names=target_names)
print ("Classification report of depth 3")
print(report)

fig, ax = plt.subplots(figsize=(12, 12))

# Plot the decision tree with custom styling
tree.plot_tree(train, ax=ax, filled=True, rounded=True, feature_names=feature_names, class_names=class_names, fontsize=8)

# Customize the plot aesthetics
ax.set_title("Decision Tree - Wine Dataset for max depth=3 (using entropy)", fontsize=20)
ax.set_xlabel("Features", fontsize=14)
ax.set_ylabel("Wine Class", fontsize=14)
# Adjust the arrow properties
for arrow in ax.get_xticklines():
    arrow.set_markersize(5)

# Adjust the spacing between subplots if needed
plt.subplots_adjust(wspace=0.5, hspace=0.5)
# Display the plot
plt.show()

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0,max_depth = 2)

train=classifier.fit(x_train, y_train)

Pred=classifier.predict(x_test)
conf = confusion_matrix(y_test, Pred)
conf

sns.heatmap(conf, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (depth 2)')
plt.show()

target_names = ['0', '1','2']
report = classification_report(y_test, Pred, target_names=target_names)
print ("Classification report of depth 2")
print(report)
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the decision tree with custom styling
tree.plot_tree(train, ax=ax, filled=True, rounded=True, feature_names=feature_names, class_names=class_names, fontsize=8)

# Customize the plot aesthetics
ax.set_title("Decision Tree - Wine Dataset for max depth=2 (using entropy)", fontsize=20)
ax.set_xlabel("Features", fontsize=14)
ax.set_ylabel("Wine Class", fontsize=14)
# Adjust the arrow properties
for arrow in ax.get_xticklines():
    arrow.set_markersize(5)

# Adjust the spacing between subplots if needed
plt.subplots_adjust(wspace=0.5, hspace=0.5)
# Display the plot
plt.show()