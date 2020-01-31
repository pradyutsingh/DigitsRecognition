import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# this is the digits dataset imported from sklearn 
digits = datasets.load_digits()

#first we want to display the forst 4 images thats why to visualise the results
#we use subplots which will return 2 values out of which one is assigned to axes.
#then we start the for loop
# step 1 -- images_and_labels is a list of zipped vectors images and labels of the dataset
# step 2 -- set_axis_off turns of the axes
#step3 -- imshow shows the images with the gray colormap and interpolation = 'nearest'
# will result an image in which pixels are displayed as a square of multiple pixels.
# step4-- set_title will set the titles for the image plots
_, axes = plt.subplots(2, 4)
images_and_labels = list(zip(digits.images, digits.target))
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

# to apply the classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
predicted = classifier.predict(X_test)

_, axes = plt.subplots(2, 4)
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)

print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")   
print("Confusion matrix is :\n%s" % disp.confusion_matrix)

plt.show()