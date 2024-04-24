import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math

use_exclusive_ratios = False

def delete_all_in_folder(folder_path):
    file_list = os.listdir(folder_path)

    # Iterate over each file and delete it
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)

def get_pca_data(n_components=5, display_eigen = False, display_data = False):
    """
        Perform Principal Component Analysis (PCA) on the provided data and generate visualizations of the PCA results.

        Parameters:
        - n_components (int): Number of principal components to retain in the PCA. Default is 5.
        - display_eigen(bool): Marker to indicate if eigenvectors should be displayed.
        - display_data(bool): Marker to indicate if sample distribution should be displayed.

        Explanation:
        - Data Preparation:
          - Deletes all files in the "processing_results\\new_pcas\\" folder.
          - Loads data from files in the "processing_results\\ratios\\" directory.
          - Standardizes data using StandardScaler.

        - Principal Component Analysis (PCA):
          - Performs PCA with the specified number of components.
          - Prints the explained variance ratio for each component and the cumulative sum.
          - Computes eigenvectors and saves them to a file.
          - Transforms data to principal components and saves them to files.

        - Visualization:
          - Generates scatter plots of principal components for different combinations.
          - Colors data points by target ('N' and 'R') and adds annotations for file names.
          - Plots arrows to indicate feature contributions to principal components.
          - Saves plots as image files.

        Output:
        - PCA results (eigenvectors, transformed principal components) are saved to files.
        - Scatter plot images showing data distribution in the principal component space are generated and saved.
        """
    delete_all_in_folder("processing_results\\new_pcas\\")

    directory = "processing_results\\ratios\\"
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
    x = StandardScaler().fit_transform(x)
    #print ("tamanho original: ",np.shape (x))

    pca = PCA(n_components)
    colunas = []
    for i in range(n_components):
        colunas.append('principal component ' + str(i + 1))
    principalComponents = pca.fit_transform(x)
    print("feature importance:", pca.explained_variance_ratio_)
    print("Sum =", sum(pca.explained_variance_ratio_))

    eigenvalues = pca.components_
    #print("Eigenvectors:", abs(eigenvalues))
    top_5_idx = np.argsort(abs(eigenvalues))[:,-6:]
    print(top_5_idx)
    for i in range(len(eigenvalues)):
        print ([abs(eigenvalues[i,e]) for e in top_5_idx[i]])
    ratios_labels = np.genfromtxt(directory + file_list[-1], dtype='str', delimiter=" , ")[:, 0]
    print(ratios_labels[top_5_idx])

    np.savetxt("processing_results\\new_pcas\\" + "eigenvectors.txt", eigenvalues)
    # print (principalComponents)
    #print ("Tamanho PCA: ", np.shape (principalComponents))

    np.savetxt("processing_results\\" + "PCA_todos_os_racios.txt", np.transpose(principalComponents))
    #print (principalComponents)

    '''   test = pca.fit(x)
    print( "explained_variance_ratio: ", test.explained_variance_ratio_)
    test_components = np.asarray(test.components_)
    print("Tamanho de test:", test_components.shape)
   #print(np.argmax(test_components, axis=1))
    print(np.argsort(abs(test_components[1]))+1)
    print(abs(test_components[1][np.argsort(abs(test_components[1]))]))'''

    principalDf = pd.DataFrame(data=principalComponents, columns=colunas)
    df = pd.DataFrame(y, columns=['target'])
    finalDf = pd.concat([principalDf, df[['target']]], axis=1)

    for x in range(n_components):
        for y in range(n_components):

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('principal component ' + str(x + 1), fontsize=15)
            ax.set_ylabel('principal component ' + str(y + 1), fontsize=15)
            ax.set_title(str(n_components)+' components PCA', fontsize=20)
            targets = ['N', 'R']
            colors = ['b', 'r']


            score = principalComponents
            coeff = np.transpose(pca.components_[[x,y], :])
            n = coeff.shape[0]
            labels = None
            labels = np.genfromtxt(directory + file_list[-1], dtype='str', delimiter=" , ")[:, 0]
            xs = score[:, x]
            ys = score[:, y]
            n = coeff.shape[0]
            scalex = 1.0 / (xs.max() - xs.min())
            scaley = 1.0 / (ys.max() - ys.min())

            if display_data:
                for target, color in zip(targets, colors):
                    indicesToKeep = finalDf['target'] == target
                    ax.scatter(finalDf.loc[indicesToKeep, 'principal component ' + str(x + 1)] * scalex
                               , finalDf.loc[indicesToKeep, 'principal component ' + str(y + 1)] * scaley
                               , c=color
                               , s=50)

                annotations = file_list[2:]
                x_plot = finalDf.loc[:, 'principal component ' + str(x + 1)] * scalex
                y_plot = finalDf.loc[:, 'principal component ' + str(y + 1)] * scaley
                for i, txt in enumerate(annotations):
                    indicesToKeep = finalDf
                    ax.annotate(txt[:-10], (x_plot[i], y_plot[i]))
                ax.legend(targets)
                ax.grid()

            if display_eigen:
                for i in range(n):
                    if math.sqrt((coeff[i, 0]*coeff[i, 0]) + (coeff[i, 1]*coeff[i, 1]))>=0.165:
                        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
                        if labels is None:
                            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center',
                                 va='center')
                        else:
                            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, str(labels[i]), color='g', ha='center', va='center')


            fig.savefig('processing_results\\new_pcas\\PCA '+ str(use_exclusive_ratios) +str(n_components)+ ' ' + str(x + 1) + ' ' + str(y + 1), bbox_inches='tight')
            plt.close(fig)
            fig.clf()


'''for i in range (4):
    get_pca_data(i+2)'''

get_pca_data (5, display_eigen=False, display_data=True)
