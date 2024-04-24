import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

use_exclusive_ratios = False
number_of_samples = 0


def get_data(directory):
    """
    Load data from a directory.

    Args:
    - directory: The directory path where the data files are located.
    - use_exclusive_ratios: Flag indicating whether to use exclusive ratios.

    Returns:
    - x: Features data.
    - y: Labels data.
    - file_list: List of file names.
    """
    file_list = os.listdir(directory)
    important_ratios = np.genfromtxt(directory + 'largest_peak_ratios_mean.txt', dtype='S')[:, 0].astype(int)

    if use_exclusive_ratios:
        x = np.zeros([len(file_list) - 2, np.shape(important_ratios)[0]])
    else:
        number_of_ratios = np.shape(np.genfromtxt(directory + file_list[-1], dtype='S', delimiter=" , "))[0]
        x = np.zeros([len(file_list) - 2, number_of_ratios])
    y = np.zeros([len(file_list) - 2, 1]).astype(str)

    counter = 0
    for i in range(len(file_list)):
        if file_list[i][0] == 'N' or file_list[i][0] == 'R':
            data = np.genfromtxt(directory + file_list[i], dtype='S', delimiter=" , ")
            if use_exclusive_ratios:
                x[counter] = data[important_ratios, 1].astype(float)
            else:
                x[counter] = data[:, 1].astype(float)
            y[counter] = file_list[i][0]
            counter += 1
    # x = StandardScaler().fit_transform(x)
    # global number_of_samples
    # number_of_samples = len(file_list) - 2
    return x, y, file_list


def fit_scaler(data):
    scaler = StandardScaler()
    x = scaler.fit_transform(data)
    return x, scaler


def apply_scaler(data, scaler):
    x = scaler.transform(data)
    return x


def fit_PCA(data, n_components=5):
    pca = PCA(n_components)
    principalComponents = pca.fit_transform(data)

    return pca, principalComponents


def apply_PCA(data, pca):
    principalComponents = pca.transform(data)

    return principalComponents


def plot_PCA(keys, principalComponents, file_list, path, n_components=5):
    """
    Plot PCA results.

    Args:
    - keys: Array containing target labels.
    - principalComponents: Principal components array.
    - file_list: List of file names.
    - path: Path to save the plots.
    - n_components: Number of principal components to plot.

    Returns:
    - None
    """

    colunas = []
    for i in range(n_components):
        colunas.append('principal component ' + str(i + 1))

    principalDf = pd.DataFrame(data=principalComponents, columns=colunas)
    df = pd.DataFrame(keys, columns=['target'])
    finalDf = pd.concat([principalDf, df[['target']]], axis=1)

    for x in range(n_components):
        for y in range(n_components):

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('principal component ' + str(x + 1), fontsize=15)
            ax.set_ylabel('principal component ' + str(y + 1), fontsize=15)
            ax.set_title(str(n_components) + ' components PCA', fontsize=20)
            targets = ['N', 'R']
            colors = ['b', 'r']
            for target, color in zip(targets, colors):
                indicesToKeep = finalDf['target'] == target
                ax.scatter(finalDf.loc[indicesToKeep, 'principal component ' + str(x + 1)]
                           , finalDf.loc[indicesToKeep, 'principal component ' + str(y + 1)]
                           , c=color
                           , s=50)

            annotations = file_list[2:]
            x_plot = finalDf.loc[:, 'principal component ' + str(x + 1)]
            y_plot = finalDf.loc[:, 'principal component ' + str(y + 1)]
            for i, txt in enumerate(annotations):
                indicesToKeep = finalDf
                ax.annotate(txt[:-10], (x_plot[i], y_plot[i]))

            ax.legend(targets)
            ax.grid()
            fig.savefig('processing_results\\classification\\' + path + '_PCA\\' + str(use_exclusive_ratios) + str(
                n_components) + ' ' + str(x + 1) + ' ' + str(y + 1), bbox_inches='tight')
            fig.clf()


def train_svm_classifier(x, y, cross_val=False, c=1, rs=0):
    classifier = svm.LinearSVC(C=c, random_state=rs)
    classifier.fit(x, y)
    if cross_val:
        scores = cross_val_score(classifier, x, y, cv=LeaveOneOut())
        # print(scores)
        # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        # print("for a c of %0.2f ,crossvalidation got %0.2f accuracy with a standard deviation of %0.2f" % (c, scores.mean(), scores.std()))

    return classifier, scores


def test_classifier(classifier, x, y):
    y_predict = classifier.predict(x)
    cm = confusion_matrix(y, y_predict)
    # print(cm)
    # y = np.ravel(y)
    accuracy = accuracy_score(np.squeeze(y), y_predict)
    # print("this is accuracy", accuracy)
    return accuracy, cm


'''for i in range (4):
    get_pca_data(i+2)'''


def svm_test_all_data(use_PCA=True, all_c=True):
    """
    Test function for training and testing classifier using data on train/test directories.
    Applies scaler, can choose to use PCA components(all or best)

    Args:
    - use_PCA: Whether to apply PCA before training the SVM classifier. Defaults to True.
    - all_c: Whether to use all classes (N and R) for training and testing. Defaults to True.

    Returns:
    - None
    """
    train_data_x, train_data_y, train_data_file_list = get_data("processing_results\\classification\\ratios_train\\")
    train_data_x, scaler = fit_scaler(train_data_x)
    test_data_x, test_data_y, test_data_file_list = get_data("processing_results\\classification\\ratios_test\\")
    test_data_x = apply_scaler(test_data_x, scaler)
    if use_PCA:
        PCA, train_data_x = fit_PCA(train_data_x, 5)
        # plot_PCA(train_data_y,train_data_x,train_data_file_list, 'train')
        test_data_x = apply_PCA(test_data_x, PCA)
        if not all_c:
            train_data_x = train_data_x[:, [0, 2, 3]]
            test_data_x = test_data_x[:, [0, 2, 3]]
            # print(train_data_x.shape, test_data_x.shape)
        # plot_PCA(test_data_y,test_data_pc,test_data_file_list, 'test')

    classifier = train_svm_classifier(train_data_x, np.ravel(train_data_y))
    test_classifier(classifier, test_data_x, np.ravel(test_data_y))


def svm_test_best_data(use_PCA=True, c=1, rs=0, ratios_path="processing_results\\ratios\\"):
    #same but splits inside function. Can use PCA components(only all)
    data_x, data_y, data_file_list = get_data(ratios_path)
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data_x, data_y, test_size=0.25,
                                                                            random_state=rs, stratify=data_y)
    # test_size = 0.25

    train_data_x, scaler = fit_scaler(train_data_x)
    test_data_x = apply_scaler(test_data_x, scaler)

    if use_PCA:
        PCA, train_data_x = fit_PCA(train_data_x, 5)
        # plot_PCA(train_data_y,train_data_x,train_data_file_list, 'train')
        test_data_x = apply_PCA(test_data_x, PCA)
        # plot_PCA(test_data_y,test_data_x,test_data_file_list, 'test')
    classifier, scores = train_svm_classifier(train_data_x, np.ravel(train_data_y), cross_val=True, c=c, rs=rs)
    acc, cm = test_classifier(classifier, test_data_x, np.ravel(test_data_y))
    return acc, cm, scores


def svm_selecting_train_data(hn, cn, use_PCA=True, c=1,
                             rs=0,
                             ratios_path="processing_results\\ratios\\"):  # hn = healthy sample number, cn = cancerous sample number
    data_x, data_y, data_file_list = get_data(ratios_path)
    """
    Allows for selecting a pair of samples for testing.
    Trains with remaining and tests with chosen.
    Can choose to use PCA components.

    Args:
    - hn: Index of the healthy sample in the dataset.
    - cn: Index of the cancerous sample in the dataset.
    - use_PCA: Whether to apply PCA before training the SVM classifier. Defaults to True.
    - c: Regularization parameter for SVM. Defaults to 1.
    - rs: Random state for data splitting. Defaults to 0.
    - ratios_path: Path to the directory containing ratio data files. Defaults to "processing_results\\ratios\\".

    Returns:
    - acc: Accuracy of the SVM classifier on the test data.
    - cm: Confusion matrix of the SVM classifier on the test data.
    - scores: Cross-validation scores of the SVM classifier.
    """

    train_index = range(len(data_x))
    test_index = [hn, cn]
    train_index.remove(hn)
    train_index.remove(cn)
    train_data_x = np.array(data_x)
    test_data_x = np.array(data_x)
    train_data_y = np.array(data_y)
    test_data_y = np.array(data_y)
    train_data_x = np.delete(train_data_x, test_index, 0)
    train_data_y = np.delete(train_data_y, test_index, 0)
    test_data_x = np.delete(test_data_x, train_index, 0)
    test_data_y = np.delete(test_data_y, train_index, 0)
    print("test_data_x")
    print(test_data_x.shape)
    print("test_data_y")
    print(test_data_y.shape)
    print("train_data_x")
    print(train_data_x.shape)
    print("train_data_y")
    print(train_data_y.shape)

    train_data_x, scaler = fit_scaler(train_data_x)
    test_data_x = apply_scaler(test_data_x, scaler)

    if use_PCA:
        PCA, train_data_x = fit_PCA(train_data_x, 5)
        # plot_PCA(train_data_y,train_data_x,train_data_file_list, 'train')
        test_data_x = apply_PCA(test_data_x, PCA)
        # plot_PCA(test_data_y,test_data_x,test_data_file_list, 'test')
    classifier, scores = train_svm_classifier(train_data_x, np.ravel(train_data_y), cross_val=True, c=c, rs=rs)
    acc, cm = test_classifier(classifier, test_data_x, np.ravel(test_data_y))
    return acc, cm, scores


def svm_find_parameters(use_PCA=True, ratios_path="processing_results\\ratios\\"):
    #Find the optimal value for the regularization parameter C in SVM classifier.
    data_x, data_y, data_file_list = get_data(ratios_path)
    data_x, scaler = fit_scaler(data_x)

    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data_x, data_y, test_size=0.25,
                                                                            random_state=0)
    if use_PCA:
        PCA, train_data_x = fit_PCA(train_data_x, 5)
        # plot_PCA(train_data_y,train_data_x,train_data_file_list, 'train')
        test_data_x = apply_PCA(test_data_x, PCA)
        # plot_PCA(test_data_y,test_data_x,test_data_file_list, 'test')
    c = 0.1
    while c < 5:
        classifier = train_svm_classifier(train_data_x, np.ravel(train_data_y), cross_val=True, c=c)
        c += 0.1


def onehundred_iterations(ratios_path="processing_results\\ratios\\"):
    """
    Perform 100 iterations of SVM classification with and without PCA.
    Compares between the two

    Args:
    - ratios_path: Path to the directory containing ratio data files. Defaults to "processing_results\\ratios\\".
    """
    No_pca_total_score = 0
    with_pca_total_score = 0
    iterations = 0
    pca_scores = []
    No_pca_scores = []
    pca_cm_mean = np.array([[0, 0], [0, 0]])
    No_pca_cm_mean = np.array([[0, 0], [0, 0]])
    c = 1
    for i in range(100000):
        print("random state " + str(i) + ":")
        PCA_acc, PCA_cm, PCA_scores = svm_selecting_train_data(c=c, use_PCA=True, rs=i, ratios_path=ratios_path)
        No_PCA_acc, No_PCA_cm, No_PCA_scores = svm_selecting_train_data(c=c, use_PCA=False, rs=i,
                                                                        ratios_path=ratios_path)
        with_pca_total_score += PCA_acc
        pca_cm_mean += PCA_cm
        No_pca_total_score += No_PCA_acc
        No_pca_cm_mean += No_PCA_cm
        iterations += 1

    No_pca_mean = No_pca_total_score / iterations
    No_pca_cm_mean = No_pca_cm_mean
    with_pca_mean = with_pca_total_score / iterations
    pca_cm_mean = pca_cm_mean

    TP = No_pca_cm_mean[1, 1]  # true positive
    TN = No_pca_cm_mean[0, 0]  # true negative
    FP = No_pca_cm_mean[0, 1]  # false positive
    FN = No_pca_cm_mean[1, 0]  # false negative
    total = TP + TN + FP + FN

    No_pca_Accuracy = (TN + TP) / total
    No_pca_Specificity = TN / (TN + FP)
    No_pca_Sensitivity = TP / (FN + TP)

    print("No PCA cm\n", No_pca_cm_mean, "\nNo PCA acc", No_pca_mean, "No PCA Specificity", No_pca_Specificity,
          "No PCA Sensitivity", No_pca_Sensitivity)
    print("With PCA cm\n", pca_cm_mean, "\nWith PCA acc", with_pca_mean)


def one_iteration(ratios_path="processing_results\\ratios\\"):
    #same but only one iteration
    random_state = 0
    c = 1
    PCA_acc, PCA_cm, PCA_scores = svm_test_best_data(c=c, use_PCA=True, rs=random_state, ratios_path=ratios_path)
    acc, cm, scores = svm_test_best_data(c=c, use_PCA=False, rs=random_state, ratios_path=ratios_path)

    TP = cm[1, 1]  # true positive
    TN = cm[0, 0]  # true negative
    FP = cm[0, 1]  # false positive
    FN = cm[1, 0]  # false negative
    total = TP + TN + FP + FN

    Accuracy = (TN + TP) / total
    Specificity = TN / (TN + FP)
    Sensitivity = TP / (FN + TP)

    PCA_TP = PCA_cm[1, 1]  # true positive
    PCA_TN = PCA_cm[0, 0]  # true negative
    PCA_FP = PCA_cm[0, 1]  # false positive
    PCA_FN = PCA_cm[1, 0]  # false negative
    PCA_total = TP + TN + FP + FN

    PCA_Accuracy = (PCA_TN + PCA_TP) / PCA_total
    PCA_Specificity = PCA_TN / (PCA_TN + PCA_FP)
    PCA_Sensitivity = PCA_TP / (PCA_FN + PCA_TP)

    print("No PCA")
    print("Cross Validation Scores : ", scores, "    Accuracy", scores.mean(), "    Standard deviation", scores.std())
    print("Confusion Matrix", "\n", cm, "\n", "Accuracy:", Accuracy, "Specificity:", Specificity, "Sensitivity:",
          Sensitivity)
    print("\n")
    print("With PCA")
    print("Cross Validation Scores : ", PCA_scores, "    Accuracy", PCA_scores.mean(), "    Standard deviation",
          PCA_scores.std())
    print("PCA Confusion Matrix", "\n", PCA_cm, "\n", "PCA Accuracy:", PCA_Accuracy, "PCA Specificity:",
          PCA_Specificity, "PCA Sensitivity:", PCA_Sensitivity)


def no_repeat_test(ratios_path="processing_results\\ratios\\"):
    # Perform SVM classification without repeating test samples.
    No_pca_total_score = 0
    with_pca_total_score = 0
    iterations = 0
    pca_scores = []
    No_pca_scores = []
    pca_cm_mean = np.array([[0, 0], [0, 0]])
    No_pca_cm_mean = np.array([[0, 0], [0, 0]])
    c = 1
    get_data(ratios_path)
    for i in range(int(number_of_samples / 2)):
        for j in range(int(number_of_samples / 2), number_of_samples):
            print("testing with sample H", i, "and C", j)
            PCA_acc, PCA_cm, PCA_scores = svm_selecting_train_data(c=c, use_PCA=True, rs=i, hn=i, cn=j,
                                                                   ratios_path=ratios_path)
            No_PCA_acc, No_PCA_cm, No_PCA_scores = svm_selecting_train_data(c=c, use_PCA=False, rs=i, hn=i, cn=j,
                                                                            ratios_path=ratios_path)
            with_pca_total_score += PCA_acc
            pca_cm_mean += PCA_cm
            No_pca_total_score += No_PCA_acc
            No_pca_cm_mean += No_PCA_cm
            iterations += 1

    No_pca_mean = No_pca_total_score / iterations
    No_pca_cm_mean = No_pca_cm_mean
    with_pca_mean = with_pca_total_score / iterations
    pca_cm_mean = pca_cm_mean

    TP = No_pca_cm_mean[1, 1]  # true positive
    TN = No_pca_cm_mean[0, 0]  # true negative
    FP = No_pca_cm_mean[0, 1]  # false positive
    FN = No_pca_cm_mean[1, 0]  # false negative
    total = TP + TN + FP + FN

    No_pca_Accuracy = (TN + TP) / total
    No_pca_Specificity = TN / (TN + FP)
    No_pca_Sensitivity = TP / (FN + TP)

    print("No PCA cm\n", No_pca_cm_mean, "\nNo PCA acc", No_pca_mean, "No PCA Specificity", No_pca_Specificity,
          "No PCA Sensitivity", No_pca_Sensitivity)
    print("With PCA cm\n", pca_cm_mean, "\nWith PCA acc", with_pca_mean)


def data_iterator(data_x, data_y):
    """
    Generate pairs of indices for iterating over data samples.

    Args:
    - data_x: Input data.
    - data_y: Target labels.

    Returns:
    - list_of_pairs: List of pairs of indices.
    """
    number_of_samples = len(data_x)
    list_of_pairs = []
    for hn in range(int(number_of_samples / 2)):
        for cn in range(int(number_of_samples / 2), number_of_samples):
            list_of_pairs.append([hn, cn])
    return list_of_pairs


def train_test_data_selector(data_x, data_y, hn, cn):  # hn = healthy sample number, cn = cancerous sample number
    """
    Split data into train and test sets based on the indices of healthy and cancerous samples.

    Args:
    - data_x: Input data.
    - data_y: Target labels.
    - hn: Index of the healthy sample.
    - cn: Index of the cancerous sample.

    Returns:
    - train_data_x: Training input data.
    - train_data_y: Training target labels.
    - test_data_x: Testing input data.
    - test_data_y: Testing target labels.
    """
    train_index = [*range(len(data_x))]
    test_index = [hn, cn]
    train_index.remove(hn)
    train_index.remove(cn)
    train_data_x = np.array(data_x)
    test_data_x = np.array(data_x)
    train_data_y = np.array(data_y)
    test_data_y = np.array(data_y)
    train_data_x = np.delete(train_data_x, test_index, 0)
    train_data_y = np.delete(train_data_y, test_index, 0)
    test_data_x = np.delete(test_data_x, train_index, 0)
    test_data_y = np.delete(test_data_y, train_index, 0)
    # print(test_data_x.shape)
    # print(test_data_y.shape)
    # print(train_data_x.shape)
    # print(train_data_y.shape)
    return train_data_x, train_data_y, test_data_x, test_data_y


def custom_leave_one_out(train_data_x, train_data_y, model, Kernel='rbf', c=1, Degree=3, Gamma='scale', Coef0=0.0,
                         apply_sca=True, apply_pca=True,
                         rs=0):  # Itera train_data removendo uma amostra cada iteração para usar como teste
    """
    Leave-one-out cross-validation for model evaluation.

    Args:
    - train_data_x: Input data for training.
    - train_data_y: Target labels for training.
    - model: Type of SVM model. Supported values: "LinearSVC", "SVC".
    - Kernel: Kernel function for SVC model. Supported values: "linear", "poly", "rbf", "sigmoid".
    - c: Regularization parameter.
    - Degree: Degree of the polynomial kernel function (for "poly" kernel).
    - Gamma: Kernel coefficient (for "poly", "rbf", "sigmoid" kernels).
    - Coef0: Independent term in the polynomial kernel function (for "poly" and "sigmoid" kernels).
    - apply_sca: Whether to apply feature scaling.
    - apply_pca: Whether to apply PCA.
    - rs: Random state for reproducibility.

    Returns:
    - mean_score: Mean accuracy score of the model.
    """
    train_index = range(len(train_data_x))

    score_list = []
    parameters_list = []

    for i in train_index:
        new_train_data_x = np.delete(train_data_x, i, 0)
        new_train_data_y = np.delete(train_data_y, i, 0)
        test_data_x = np.array(train_data_x[i])
        test_data_y = np.array(train_data_y[i])

        test_data_x = test_data_x.reshape(1, -1)
        # fazer_scaler
        if apply_sca:
            new_train_data_x, new_scaler = fit_scaler(new_train_data_x)
            test_data_x = apply_scaler(test_data_x, new_scaler)
        # fazer_pca
        if apply_pca:
            new_pca, new_train_data_x = fit_PCA(new_train_data_x)
            test_data_x = apply_PCA(test_data_x, new_pca)

        # test_data_x = test_data_x.ravel
        # train_svm_classifier(new_train_data_x,new_train_data_y)
        if model == "LinearSVC":
            classifier = svm.LinearSVC(C=c, random_state=rs)
        elif model == "SVC":
            if Kernel == 'linear':
                classifier = svm.SVC(C=c, kernel=Kernel)
            elif Kernel == 'poly':
                classifier = svm.SVC(C=c, kernel=Kernel, degree=Degree, gamma=Gamma, coef0=Coef0)
            elif Kernel == 'rbf':
                classifier = svm.SVC(C=c, kernel=Kernel, degree=Degree, gamma=Gamma)
            elif Kernel == 'sigmoid':
                classifier = svm.SVC(C=c, kernel=Kernel, gamma=Gamma, coef0=Coef0)
            else:
                print('invalid kernel for custom_leave_one_out')
                exit()
        else:
            print('invalid model for custom_leave_one_out')
            exit()
        classifier.fit(new_train_data_x, np.ravel(new_train_data_y))
        # classifier = train_svm_classifier(new_train_data_x,new_train_data_y, c)
        y_predict = classifier.predict(test_data_x)
        if np.squeeze(y_predict) == np.squeeze(test_data_y):
            score_list.append(1)
        else:
            score_list.append(0)
        mean_score = np.mean(np.array(score_list))

    return mean_score


def custom_grid_search(models=['LinearSVC', 'SVC'], kernel=['linear', 'poly', 'rbf', 'sigmoid'], c=[-5, 16, 0.5],
                       degree=6, gamma=[-15, 4, 0.5], Coef0=[0, 0, 1]):
    """
    Perform a custom grid search over SVM model parameters.

    Args:
    - models: List of SVM models to search. Supported values: ['LinearSVC', 'SVC'].
    - kernel: List of kernel functions for SVC model. Supported values: ['linear', 'poly', 'rbf', 'sigmoid'].
    - c: List specifying the start, end, and interval for the regularization parameter.
    - degree: Maximum degree of the polynomial kernel function (for "poly" kernel).
    - gamma: List specifying the start, end, and interval for the kernel coefficient (for "poly", "rbf", "sigmoid" kernels).
    - Coef0: List specifying the start, end, and interval for the independent term in the kernel function (for "poly" and "sigmoid" kernels).

    Returns:
    - parameter_list: List of lists containing parameter combinations for grid search.
    """
    parameter_list = []

    c_start, c_end, c_interval = c[0], c[1], c[2]
    c_values = np.around(2 ** np.arange(c_start, c_end, c_interval, dtype=float), 14)

    degree_values = np.arange(1, degree)

    gamma_start, gamma_end, gamma_interval = gamma[0], gamma[1], gamma[2]
    gamma_values = np.around(2 ** np.arange(gamma_start, gamma_end, gamma_interval, dtype=float), 14)

    Coef0_start, Coef0_end, Coef0_interval = Coef0[0], Coef0[1], Coef0[2]
    Coef0_values = np.around(np.arange(Coef0_start, Coef0_end, Coef0_interval), 14)

    if len(Coef0_values) == 0: Coef0_values = [0]

    for i in models:
        if i == 'LinearSVC':
            for j in c_values:
                parameter_list.append([i, 'default', j, 0, 0, 0])
        elif i == 'SVC':
            for k in kernel:
                if k == 'linear':
                    for j in c_values:
                        parameter_list.append([i, k, j, 0, 0, 0])
                elif k == 'poly':
                    for j in c_values:
                        for d in degree_values:
                            for g in gamma_values:
                                for t in Coef0_values:
                                    parameter_list.append([i, k, j, d, g, t])
                elif k == 'rbf':
                    for j in c_values:
                        for g in gamma_values:
                            parameter_list.append([i, k, j, 0, g, 0])
                elif k == 'sigmoid':
                    for j in c_values:
                        for g in gamma_values:
                            for t in Coef0_values:
                                parameter_list.append([i, k, j, 0, g, t])
                else:
                    print('invalid kernel for custom_grid_search')
                    exit()
        else:
            print('invalid model for custom_grid_search')
            exit()

    # print(c_values, degree_values, gamma_values, Coef0_values)

    return parameter_list


def save_parameters_list(parameters_list, filepath):
    parameters_list = np.array(parameters_list)
    np.savetxt(filepath, parameters_list, fmt="%s")


def load_parameters_list(filepath):
    parameter_list = np.genfromtxt(filepath, dtype=["U32", "U32", float, int, float, float])
    return parameter_list


def save_result_list(result_list, filepath):
    np.savetxt(filepath, result_list, fmt="%s")


def load_result_list(filepath):
    result_list = np.genfromtxt(filepath, dtype=["U32", "U32", float, int, float, float, int, int, float])
    return result_list


def save_non_uniform_array(array, filepath):
    with open(filepath, 'w') as file:
        for row in array:
            for sub_row in row:
                sub_row_str = ' '.join(map(str, sub_row))
                file.write(sub_row_str + '\n')
            file.write('\n')


def possibility_iterator(parameters_to_test, apply_pca=True, apply_sca=True,
                         ratios_path="processing_results\\ratios\\"):
    """
    Iterate over all possible pairs of samples and test different parameter combinations for SVM models using custom leave-one-out cross-validation.

    Args:
    - parameters_to_test: List of parameter combinations to test, obtained from custom_grid_search.
    - apply_pca: Boolean indicating whether to apply PCA to the data.
    - apply_sca: Boolean indicating whether to apply standard scaling to the data.
    - ratios_path: Path to the directory containing the data.

    Returns:
    - scores_list: List of all scores obtained for each parameter combination and pair of samples.
    - top_scores: List of top scores obtained for each pair of samples, regardless of kernel type.
    - top_scores_by_kernel: List of top scores obtained for each pair of samples, grouped by kernel type.
    """
    data_x, data_y, data_file_list = get_data(ratios_path)
    list_of_pairs = data_iterator(data_x, data_y)
    scores_list = []
    kernel_types = []
    total = len(list_of_pairs) * len(parameters_to_test)
    count = 0
    for i in list_of_pairs:
        hn, cn = i
        train_data_x, train_data_y, test_data_x, test_data_y = train_test_data_selector(data_x, data_y, hn, cn)
        for j in parameters_to_test:
            score = custom_leave_one_out(train_data_x, train_data_y, j[0], j[1], j[2], j[3], j[4], j[5],
                                         apply_pca=apply_pca, apply_sca=apply_sca)
            label = list(j.tolist()).copy()
            label.append(' result for pair ' + str(hn) + ' and ' + str(cn) + ' = ' + str(score))
            score_append = list(j.tolist()).copy()
            score_append.append(hn)
            score_append.append(cn)
            score_append.append(score)
            scores_list.append(score_append)
            # print(label)

            if j[1] not in kernel_types:
                kernel_types.append(j[1])
            count += 1
            print("Grid search at " + str(round((count / total) * 100, 2)) + "%")

    scores_list = np.array(scores_list, dtype=object)
    top_scores = []
    top_scores_by_kernel = []
    for i in list_of_pairs:
        hn, cn = i
        hn_iterations = scores_list[scores_list[:, 6] == hn, :]
        cn_iterations = hn_iterations[hn_iterations[:, 7] == cn, :]
        i_max_number = np.amax(cn_iterations, axis=0)[8]
        i_best_iterations = cn_iterations[cn_iterations[:, 8] == i_max_number, :]
        for t in i_best_iterations:
            top_scores.append(t)
        for j in kernel_types:
            j_iterations = cn_iterations[cn_iterations[:, 1] == j, :]
            j_max_number = np.amax(j_iterations, axis=0)[8]
            j_best_iterations = j_iterations[j_iterations[:, 8] == j_max_number, :]
            for k in j_best_iterations:
                top_scores_by_kernel.append(k)
            # print (j_best_iterations)
    top_scores = np.array(top_scores, dtype=object)
    top_scores_by_kernel = np.array(top_scores_by_kernel, dtype=object)

    return scores_list, top_scores, top_scores_by_kernel


# onehundred_iterations()
# one_iteration()
# no_repeat_test()


def update_parameter_list(models=['LinearSVC', 'SVC'], kernel=['linear', 'poly', 'rbf', 'sigmoid'], c=[-5, 16, 0.5],
                          degree=6, gamma=[-15, 4, 0.5], Coef0=[0, 0, 1]):
    """
    Update the list of parameter combinations for grid search and save it to a file.

    Args:
    - models: List of model types to include in the grid search.
    - kernel: List of kernel types to include in the grid search.
    - c: List specifying the range of values for the regularization parameter C.
    - degree: Integer specifying the range of degrees for polynomial kernels.
    - gamma: List specifying the range of values for the gamma parameter.
    - Coef0: List specifying the range of values for the coef0 parameter.
    """
    parameters_to_test = custom_grid_search(models=models, kernel=kernel, c=c, degree=degree, gamma=gamma, Coef0=Coef0)
    save_parameters_list(parameters_to_test, "processing_results\\classification\\" + "parameters_to_test.txt")


def create_score_lists(apply_pca=True, apply_sca=True, path="processing_results\\classification\\",
                       ratios_path="processing_results\\ratios\\"):
    """
    Perform grid search with custom leave-one-out cross-validation using specified parameter combinations and save the results.

    Args:
    - apply_pca: Boolean indicating whether to apply PCA to the data.
    - apply_sca: Boolean indicating whether to apply standard scaling to the data.
    - path: Path to the directory where the parameter combinations and scores will be saved.
    - ratios_path: Path to the directory containing the data.
    """

    parameter_list = load_parameters_list(path + "parameters_to_test.txt")
    scores_list, top_scores, top_scores_by_kernel = possibility_iterator(parameter_list, apply_pca=apply_pca,
                                                                         apply_sca=apply_sca, ratios_path=ratios_path)
    save_path = path + "scores\\"
    if apply_sca:
        save_path += "com_scaler\\"
    else:
        save_path += "sem_scaler\\"
    if apply_pca:
        save_path += "com_pca\\"
    else:
        save_path += "sem_pca\\"
    save_result_list(scores_list, save_path + "scores_list.txt")
    save_result_list(top_scores, save_path + "top_scores.txt")
    save_result_list(top_scores_by_kernel, save_path + "top_scores_by_kernel.txt")


def load_score_lists(apply_sca=True, apply_pca=True):
    """
    Load score lists from files based on specified preprocessing configurations.

    Args:
    - apply_sca: Boolean indicating whether scaling was applied to the data.
    - apply_pca: Boolean indicating whether PCA was applied to the data.

    Returns:
    - scores_list: List of scores for each parameter combination and data pair.
    - top_scores: List of top scores for each data pair.
    - top_scores_by_kernel: List of top scores grouped by kernel type for each data pair.
    """
    added_path = "processing_results\\classification\\scores\\"
    if apply_sca:
        if apply_pca:
            added_path += "com_scaler\\com_pca\\"
        else:
            added_path += "com_scaler\\sem_pca\\"
    else:
        if apply_pca:
            added_path += "sem_scaler\\com_pca\\"
        else:
            added_path += "sem_scaler\\sem_pca\\"
    scores_list = load_result_list(added_path + "scores_list.txt")
    top_scores = load_result_list(added_path + "top_scores.txt")
    top_scores_by_kernel = load_result_list(added_path + "top_scores_by_kernel.txt")
    return scores_list, top_scores, top_scores_by_kernel


def top_scores_analizer(top_scores):
    """
    Analyze the top scores to determine the correspondence between kernel types and data pairs,
    as well as count the number of appearances and the number of samples where each kernel type is best.

    Args:
    - top_scores: List of top scores for each data pair.

    Returns:
    - correspondence_list: List containing the correspondence between kernel types and data pairs.
    - best_kernel_count: List counting the number of appearances and the number of samples where each kernel type is best.
    """

    kernel_list = []
    pair_list = []
    correspondence_list = []
    best_kernel_count = [["kernel", "Number of appearences", "Number of samples where it's best"]]
    for i in top_scores:
        if [i[1], i[6], i[7]] not in kernel_list:
            kernel_list.append([i[1], i[6], i[7]])
        if [i[6], i[7]] not in pair_list:
            pair_list.append([i[6], i[7]])
        if i[1] not in np.array(best_kernel_count, dtype=object)[:, 0]:
            best_kernel_count.append([i[1], 0, 0])
        for j in best_kernel_count:
            if j[0] == i[1]:
                j[1] += 1

    for j in pair_list:
        mini_correspondence_list = []
        for i in kernel_list:
            if [i[1], i[2]] == j:
                mini_correspondence_list.append(i[0])
        correspondence_list.append([j] + [mini_correspondence_list])

    for i in correspondence_list:
        for j in i[1]:
            for k in best_kernel_count:
                if k[0] == j:
                    k[2] += 1

    return correspondence_list, best_kernel_count


def check_best_kernel(apply_sca=False, apply_pca=False):
    """
    Check the best kernel types for each data pair and the overall count of each kernel type.

    Args:
    - apply_sca: Boolean indicating whether scaling was applied.
    - apply_pca: Boolean indicating whether PCA was applied.

    Prints:
    - Best kernel types for each data pair.
    - Overall count of each kernel type.
    """
    scores_list, top_scores, top_scores_by_kernel = load_score_lists(apply_sca, apply_pca)
    best_kernels_by_pairs, best_kernel_count = top_scores_analizer(top_scores)
    for i in best_kernels_by_pairs:
        print(i)
    print("")
    print("")
    for i in best_kernel_count:
        print(i)


import matplotlib.ticker as mticker


def log_tick_formatter(val, pos=None):
    return r"$10^{:.0f}$".format(val)


def analize_parameter_space(kernel='sigmoid', apply_sca=False, apply_pca=False):
    """
    Analyzes the parameter space for Support Vector Machine (SVM) classifiers with different kernel types.

    This function loads score lists from previous classification experiments, then visualizes the parameter space
    for each data pair and kernel type using scatter plots. The scatter plots represent the relationship between
    hyperparameters such as C (regularization parameter), Gamma (kernel coefficient), and Degree (for polynomial kernel),
    with colors indicating whether the hyperparameter combination yielded top-performing models.

    Args:
    - kernel (str): The type of kernel to analyze ('linear', 'poly', 'rbf', 'sigmoid').
    - apply_sca (bool): Indicates whether scaling was applied to the data.
    - apply_pca (bool): Indicates whether Principal Component Analysis (PCA) was applied to the data.

    Saves:
    - Scatter plots visualizing the parameter space for each data pair and kernel type.
    """
    scores_list, top_scores, top_scores_by_kernel = load_score_lists(apply_sca, apply_pca)

    sigmoid_parameters = []
    total_sigmoid_parameters = []
    pair_list = []
    first_iteration = True
    for i in scores_list:
        pair = [i[6], i[7]]
        if first_iteration:
            pair_list.append(pair)
            first_iteration = False
        if i[1] == kernel:
            if kernel == 'poly':
                sigmoid_parameters.append([i[2], i[4], i[3]])
            else:
                sigmoid_parameters.append([i[2], i[4]])
        if pair not in pair_list:
            total_sigmoid_parameters.append([pair_list[-1], sigmoid_parameters])
            sigmoid_parameters = []
            pair_list.append(pair)
    total_sigmoid_parameters.append([pair_list[-1], sigmoid_parameters])
    sigmoids_top_scores = []
    for i in top_scores_by_kernel:
        if i[1] == kernel:
            sigmoids_top_scores.append(i)
    sigmoids_top_or_not = []
    # sigmoids_top_scores = np.array(sigmoids_top_scores)
    for j in total_sigmoid_parameters:
        sigmoids_top_or_not_line = []
        pair_top_scores = []
        for i in sigmoids_top_scores:
            if [i[6], i[7]] == j[0]:
                if kernel == 'poly':
                    pair_top_scores.append([i[2], i[4], i[3]])
                else:
                    pair_top_scores.append([i[2], i[4]])
        for i in j[1]:
            if i in pair_top_scores:
                sigmoids_top_or_not_line.append('red')
            else:
                sigmoids_top_or_not_line.append('green')
        sigmoids_top_or_not.append([j[0], sigmoids_top_or_not_line])
    for i in total_sigmoid_parameters:
        i[1] = np.transpose(i[1]).tolist()
        # print (i)
    # for i in sigmoids_top_or_not:
    # print (i)

    if kernel == 'poly':
        for i in range(len(sigmoids_top_or_not)):
            fig = plt.figure(figsize=(8, 8))
            ax = plt.axes(projection='3d')
            # ax.set_yscale('log')
            # ax.set_xscale('log')

            ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

            ax.set_xlabel('C', fontsize=15)
            ax.set_ylabel('Gamma', fontsize=15)
            ax.set_zlabel('Degree', fontsize=15)
            ax.set_title(str(sigmoids_top_or_not[i][0]) + " - " + kernel, fontsize=20)
            for j in range(len(total_sigmoid_parameters[i][1][0])):
                ax.scatter3D(np.log10(total_sigmoid_parameters[i][1][0][j]),
                             np.log10(total_sigmoid_parameters[i][1][1][j]), total_sigmoid_parameters[i][1][2][j],
                             c=sigmoids_top_or_not[i][1][j])
            fig.savefig('processing_results\\classification\\parameter_study\\' + kernel + " - " + str(
                sigmoids_top_or_not[i][0]),
                        bbox_inches='tight')
            fig.clf()
    elif kernel == 'linear' or kernel == 'default':
        for i in range(len(sigmoids_top_or_not)):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            # ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel('C', fontsize=15)
            ax.set_ylabel('Gamma', fontsize=15)
            ax.set_title(str(sigmoids_top_or_not[i][0]) + " - " + kernel, fontsize=20)
            for j in range(len(total_sigmoid_parameters[i][1][0])):
                ax.scatter(total_sigmoid_parameters[i][1][0][j], total_sigmoid_parameters[i][1][1][j],
                           c=sigmoids_top_or_not[i][1][j])
            fig.savefig('processing_results\\classification\\parameter_study\\' + kernel + " - " + str(
                sigmoids_top_or_not[i][0]), bbox_inches='tight')
            fig.clf()
    else:
        for i in range(len(sigmoids_top_or_not)):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel('C', fontsize=15)
            ax.set_ylabel('Gamma', fontsize=15)
            ax.set_title(str(sigmoids_top_or_not[i][0]) + " - " + kernel, fontsize=20)
            for j in range(len(total_sigmoid_parameters[i][1][0])):
                ax.scatter(total_sigmoid_parameters[i][1][0][j], total_sigmoid_parameters[i][1][1][j],
                           c=sigmoids_top_or_not[i][1][j])
            fig.savefig('processing_results\\classification\\parameter_study\\' + kernel + " - " + str(
                sigmoids_top_or_not[i][0]), bbox_inches='tight')
            fig.clf()


def choose_testing_parameters(kernel='sigmoid', apply_sca=True, apply_pca=True,
                              ratios_path="processing_results\\ratios\\"):
    """
    Selects the testing parameters for each data pair based on the top-performing models.

    This function loads the top scores obtained for each kernel type and each data pair, then chooses the testing
    parameters for each pair by selecting the hyperparameters corresponding to the best-performing model.

    Args:
    - kernel (str): The type of kernel to consider ('linear', 'poly', 'rbf', 'sigmoid').
    - apply_sca (bool): Indicates whether scaling was applied to the data.
    - apply_pca (bool): Indicates whether Principal Component Analysis (PCA) was applied to the data.
    - ratios_path (str): The path to the directory containing ratio data files.

    Returns:
    - pair_top_score (list): A list containing the selected testing parameters for each data pair.
    """
    scores_list, top_scores, top_scores_by_kernel = load_score_lists(apply_sca, apply_pca)
    data_x, data_y, data_file_list = get_data(ratios_path)
    pair_list = []
    for i in top_scores:
        if [i[6], i[7]] not in pair_list:
            pair_list.append([i[6], i[7]])
    pair_top_score = []
    kernel_top_scores = []
    for pair in pair_list:
        best_scores_by_pair = []
        for j in top_scores_by_kernel:
            if j[1] == kernel and pair == [j[6], j[7]]:
                best_scores_by_pair.append(j)
        kernel_top_scores.append([pair, best_scores_by_pair])
    for i in kernel_top_scores:
        hn, cn = i[0]
        train_data_x, train_data_y, test_data_x, test_data_y = train_test_data_selector(data_x, data_y, hn, cn)
        if apply_sca:
            train_data_x, new_scaler = fit_scaler(train_data_x)
        # fazer_pca
        if apply_pca:
            new_pca, train_data_x = fit_PCA(train_data_x)
        x_var = np.var(train_data_x)
        scale = 1 / (len(train_data_x[0]) * x_var)
        options = i[1]
        option_scores = []
        for j in options:
            if kernel == 'default' or kernel == 'linear':
                option_score = np.abs(np.log2(j[2]))
            else:
                option_score = np.abs(np.log2(j[2])) + np.abs(np.log2(scale) - np.log2(j[4])) + degree
            option_scores.append(option_score)
        min = np.min(option_scores)
        lowest_pos = np.where(option_scores == min)
        pair_top_score.append([i[0], options[lowest_pos[0][0]]])

    return pair_top_score


def testing_from_best_score(chosen_scores, apply_pca=True, apply_sca=True, ratios_path="processing_results\\ratios"):
    """
    Performs testing using the selected best scores obtained from the top-performing models.

    This function takes the chosen best scores for each data pair and performs testing using the specified model
    parameters. It then evaluates the performance of the model using confusion matrices and calculates mean accuracy
    and standard deviation across all tests.

    Args:
    - chosen_scores (list): A list containing the selected testing parameters for each data pair.
    - apply_pca (bool): Indicates whether Principal Component Analysis (PCA) was applied to the data.
    - apply_sca (bool): Indicates whether scaling was applied to the data.
    - ratios_path (str): The path to the directory containing ratio data files.

    Returns:
    - total_CM (array): The total confusion matrix obtained from all tests.
    - final_mean (float): The mean accuracy across all tests.
    - std (float): The standard deviation of the mean accuracy.
    """

    data_x, data_y, data_file_list = get_data(ratios_path)
    pair_results = []
    means = []
    total_CM = []
    failure_list = []
    failure_count = []
    first_iteration = True
    for i in chosen_scores:
        model, kernel, c, degree, gamma, coef0, hn, cn, cross_val_obtained = i[1]
        train_data_x, train_data_y, test_data_x, test_data_y = train_test_data_selector(data_x, data_y, hn, cn)
        # fazer_scaler
        if apply_sca:
            train_data_x, new_scaler = fit_scaler(train_data_x)
            test_data_x = apply_scaler(test_data_x, new_scaler)
        # fazer_pca
        if apply_pca:
            new_pca, train_data_x = fit_PCA(train_data_x)
            test_data_x = apply_PCA(test_data_x, new_pca)

        if model == "LinearSVC":
            classifier = svm.LinearSVC(C=c)
        elif model == "SVC":
            if kernel == 'linear':
                classifier = svm.SVC(C=c, kernel=kernel)
            elif kernel == 'poly':
                classifier = svm.SVC(C=c, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
            elif kernel == 'rbf':
                classifier = svm.SVC(C=c, kernel=kernel, gamma=gamma)
            elif kernel == 'sigmoid':
                classifier = svm.SVC(C=c, kernel=kernel, gamma=gamma, coef0=coef0)
            else:
                print('invalid kernel for testing_from_best_score')
                exit()
        else:
            print('invalid model for testing_from_best_score')
            exit()
        classifier.fit(train_data_x, np.ravel(train_data_y))
        y_predict = classifier.predict(test_data_x)
        pass_test = np.squeeze(y_predict) == np.squeeze(test_data_y)
        if not pass_test[0]:
            if hn not in failure_list:
                failure_count.append([hn, 1])
            else:
                for t in failure_count:
                    if hn == t[0]:
                        t[1] += 1
            failure_list.append(hn)
        if not pass_test[1]:
            if cn not in failure_list:
                failure_count.append([cn, 1])
            else:
                for t in failure_count:
                    if cn == t[0]:
                        t[1] += 1
            failure_list.append(cn)
        cm = confusion_matrix(np.squeeze(y_predict), np.squeeze(test_data_y), labels=['N', 'R'])
        if first_iteration:
            first_iteration = False
            total_CM = cm
        else:
            total_CM += cm
        means.append((cm[0][0] + cm[1][1]) / np.sum(cm))
    # print (failure_list)
    # print (failure_count)
    final_mean = (total_CM[0][0] + total_CM[1][1]) / np.sum(total_CM)
    std = np.std(means)
    return total_CM, final_mean, std


def get_new_parameter_limits(kernel, apply_pca=True, apply_sca=True, ratios_path="processing_results\\train\\"):
    """
    Extracts new parameter limits based on the chosen testing parameters for a specific kernel.

    This function analyzes the chosen testing parameters for a given kernel and extracts the maximum and minimum values
    for each parameter (C, degree, gamma, coef0). It then prints out the logarithmic limits for C and gamma, and the
    linear limits for degree.

    Args:
    - kernel (str): The kernel for which to extract the parameter limits.
    - apply_pca (bool): Indicates whether Principal Component Analysis (PCA) was applied to the data.
    - apply_sca (bool): Indicates whether scaling was applied to the data.
    - ratios_path (str): The path to the directory containing ratio data files.

    Returns:
    None
    """
    result = choose_testing_parameters(kernel, apply_pca=apply_pca, apply_sca=apply_sca, ratios_path=ratios_path)
    c_list = []
    degree_list = []
    gamma_list = []
    coef0_list = []
    for i in result:
        c_list.append(i[1][2])
        degree_list.append(i[1][3])
        gamma_list.append(i[1][4])
        coef0_list.append(i[1][5])
    print("c:", np.log2(np.max(c_list)), np.log2(np.min(c_list)), "degree:", np.max(degree_list), np.min(degree_list),
          "gamma:", np.log2(np.max(gamma_list)), np.log2(np.min(gamma_list)))


def test_get_matrix(kernels, apply_pca, apply_sca, ratios_path="processing_results\\train\\"):
    """
    Perform testing based on chosen parameters for multiple kernels and return the results.

    This function iterates over a list of kernels, obtains the chosen testing parameters for each kernel, performs
    testing using the obtained parameters, and collects the results. It prints the chosen parameters, confusion matrix,
    final mean accuracy, and standard deviation for each kernel.

    Args:
    - kernels (list): A list of strings representing the kernels for testing.
    - apply_pca (bool): Indicates whether Principal Component Analysis (PCA) was applied to the data.
    - apply_sca (bool): Indicates whether scaling was applied to the data.
    - ratios_path (str): The path to the directory containing ratio data files.

    Returns:
    - final_results (list): A list containing the results of testing for each kernel.
    """
    final_results = []
    for kernel in kernels:
        result = choose_testing_parameters(kernel, apply_pca=apply_pca, apply_sca=apply_sca, ratios_path=ratios_path)
        total_CM, final_mean, std = testing_from_best_score(result, apply_pca=apply_pca, apply_sca=apply_sca,
                                                            ratios_path=ratios_path)
        print(result)
        print("")
        print(total_CM, final_mean, std)
        print("")
        print("")
        # input("Press enter to continue")
        final_results.append([[result], [total_CM], [final_mean, std]])
    return (final_results)


model = ['LinearSVC', 'SVC']
Kernel = ['linear', 'poly', 'rbf', 'sigmoid']
c = [-5, 16, 0.5]
degree = 6
gamma = [-15, 4, 0.5]
Coef0 = [0, 0, 1]


# apply_pca = True
# apply_sca = False

# create_parameters()

# for i in apply_sca:
#    for j in apply_pca:
#        create_score_lists(apply_pca=j, apply_sca=i)
# check_best_kernel()

# tests = ['sigmoid']
# for i in tests:
#    analize_parameter_space(i)
# create_score_lists(apply_pca=apply_pca, apply_sca=apply_sca)
# kernels = ['default', 'linear', 'poly', 'rbf', 'sigmoid']
# test_get_matrix(kernels=kernels ,apply_pca=apply_pca, apply_sca=apply_sca)
# for i in kernels:
# get_new_parameter_limits(kernel=i, apply_pca=apply_pca, apply_sca=apply_sca)

def exploration(recreate_score_list=True, grid_models=['LinearSVC', 'SVC'],
                grid_kernel=['linear', 'poly', 'rbf', 'sigmoid'], grid_c=[-5, 16, 0.5], grid_degree=6,
                grid_gamma=[-15, 4, 0.5], grid_Coef0=[0, 0, 1], kernels=['default', 'linear', 'poly', 'rbf', 'sigmoid'],
                test_PCA=[False, True], test_SCA=[False, True],
                save_path="processing_results\\classification\\classifier_results\\",
                ratios_path="processing_results\\ratios\\"):
    """
    Explore the parameter space of classifiers.

    This function performs an exploration of the parameter space for various classifiers by iteratively testing different
    combinations of parameters, evaluating their performance, and saving the results.

    Parameters:
    - recreate_score_list (bool): If True, recreate the score list by updating the parameter list and creating score lists.
    - grid_models (list of str): List of models to be considered during grid search for parameter optimization.
    - grid_kernel (list of str): List of kernel types to be considered during grid search.
    - grid_c (list of float): List representing the range of values for the regularization parameter C during grid search.
    - grid_degree (int): Maximum degree of the polynomial kernel during grid search.
    - grid_gamma (list of float): List representing the range of values for the gamma parameter during grid search.
    - grid_Coef0 (list of float): List representing the range of values for the coef0 parameter during grid search.
    - kernels (list of str): List of kernel types to be tested.
    - test_PCA (list of bool): List representing whether to test with PCA or not.
    - test_SCA (list of bool): List representing whether to test with scaler or not.
    - save_path (str): Path to save the exploration results.
    - ratios_path (str): Path to the directory containing the data ratios.

    Returns:
    None
    """
    for SCA in test_SCA:
        for PCA in test_PCA:
            if recreate_score_list:
                update_parameter_list(models=grid_models, kernel=grid_kernel, c=grid_c, degree=grid_degree,
                                      gamma=grid_gamma, Coef0=grid_Coef0)
                create_score_lists(apply_pca=PCA, apply_sca=SCA, ratios_path=ratios_path)
            path = save_path
            if SCA:
                if PCA:
                    path += "with_scaler\\with_pca\\"
                else:
                    path += "with_scaler\\without_pca\\"
            else:
                if PCA:
                    path += "without_scaler\\with_pca\\"
                else:
                    path += "without_scaler\\without_pca\\"
            result = test_get_matrix(kernels=kernels, apply_pca=PCA, apply_sca=SCA, ratios_path=ratios_path)
            save_non_uniform_array(result, path + "kernel_parameters.txt")


exploration(recreate_score_list=True)
