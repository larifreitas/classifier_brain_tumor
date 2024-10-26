import os, cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


train_folder = './Training'
test_folder = './Testing'

def preprocessing(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(128,128)) 
    img = cv2.equalizeHist(img)
    return img

# extração das caracteristicas com  HOG
def extract_HOG_features(img):
    ft, _ = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2), visualize=True)
    return ft

# Extralçao das caracteristicas com LBP
def extract_LBP_features(img):
    ft = local_binary_pattern(img,24,8,method="uniform")
    (hist, _) = np.histogram(ft.ravel(),bins=np.arange(0,59),range=(0,58))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist


def load_data(folder_path, label):
    image_paths = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_paths.append(os.path.join(folder_path, filename))
            labels.append(label)
    return image_paths, labels

label_map = { 'glioma': 0, 'meningioma': 1, 'pituitary': 2, 'notumor': 3 }

X_train_paths = []
y_train = []
for label_name, label_value in label_map.items():
    folder_path = os.path.join(train_folder, label_name)
    paths, labels = load_data(folder_path, label_value)
    X_train_paths.extend(paths)
    y_train.extend(labels)

hog_features_train = []
lbp_features_train = []
for path in X_train_paths:
    img = preprocessing(path)
    hog_features_train.append(extract_HOG_features(img))
    lbp_features_train.append(extract_LBP_features(img))


X_test_paths = []
y_test = []
for label_name, label_value in label_map.items():
    folder_path = os.path.join(test_folder, label_name)
    paths, labels = load_data(folder_path, label_value)
    X_test_paths.extend(paths)
    y_test.extend(labels)

hog_features_test = []
lbp_features_test = []
for path in X_test_paths:
    img = preprocessing(path)
    hog_features_test.append(extract_HOG_features(img))
    lbp_features_test.append(extract_LBP_features(img))


# avaliar com HOG
svmHOG = SVC(kernel='linear')
svmHOG.fit(hog_features_train, y_train)
knnHOG=KNeighborsClassifier(n_neighbors=5)
knnHOG.fit(hog_features_train, y_train)
nbHOG = GaussianNB()
nbHOG.fit(hog_features_train, y_train)

# avaliar com LBP
svmLBP = SVC(kernel='linear')
svmLBP.fit(lbp_features_train, y_train)
knnLBP=KNeighborsClassifier(n_neighbors=5)
knnLBP.fit(lbp_features_train, y_train)
nbLBP = GaussianNB()
nbLBP.fit(lbp_features_train,y_train)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Resultado para HOG
results_hog = {"SVM": evaluate_model(svmHOG, hog_features_test, y_test),
               "KNN": evaluate_model(knnHOG, hog_features_test, y_test),
               "Naive Bayes": evaluate_model(nbHOG, hog_features_test, y_test)}

print("-----RESULTADOS COM HOG:\n")
for classifier, metrics in results_hog.items():
    print(f"---{classifier}: \naccuracy_score = {metrics[0]:.4f} | precision_score= {metrics[1]:.4f} | recall_score = {metrics[2]:.4f} | f1_score = {metrics[3]:.4f}\n")
print(f"------------------------------------------------------------------------------------------------------------------------------------------------")

# Resultados para LBP
results_lbp = {"SVM": evaluate_model(svmLBP, lbp_features_test, y_test),
               "KNN": evaluate_model(knnLBP, lbp_features_test, y_test),
               "Naive Bayes": evaluate_model(nbLBP, lbp_features_test, y_test)}


print("-----RESULTADOS COM LBP:\n")
for classifier, metrics in results_lbp.items():
    print(f"---{classifier}: \naccuracy_score = {metrics[0]:.4f} | precision_score= {metrics[1]:.4f} | recall_score = {metrics[2]:.4f} | f1_score = {metrics[3]:.4f}\n")
