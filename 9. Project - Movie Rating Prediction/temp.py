from sklearn.metrics import confusion_matrix

Y_ = mnb.predict(X)

cnf_matrix = confusion_matrix(Y, Y_)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix',cmap=plt.cm.Accent)