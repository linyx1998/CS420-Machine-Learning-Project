import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

colors = [plt.cm.tab10(i / 10.0) for i in range(25)]

def draw(berforPCA, PCA, name, title):
    print("Drawing %s" % (name))
    model = TSNE(n_components=2)
    source2D = np.ascontiguousarray(model.fit_transform(berforPCA).transpose())
    target2D = np.ascontiguousarray(model.fit_transform(PCA).transpose())
    plt.figure()
    plt.scatter(source2D[0], source2D[1], marker='o', cmap=colors[0], s=10, alpha=0.5, label='original data')
    plt.scatter(target2D[0], target2D[1], marker='^', cmap=colors[1], s=10, alpha=0.5, label='PCA rudeced')
    plt.legend()
    plt.title(title)
    plt.show()
    plt.savefig(name)

def draw_original(berforPCA, train_label, name, title):
    print("Drawing %s" % (name))
    model = TSNE(n_components=2)
    source2D = np.ascontiguousarray(model.fit_transform(berforPCA).transpose())
    plt.figure()
    for i in range(train_label.shape[0]):
        plt.scatter(source2D[0][i], source2D[1][i], marker='o', cmap=colors[train_label[i]], s=10, alpha=0.5, label='original data')
    plt.title(title)
    plt.show()
    plt.savefig(name)

def draw_PCA(berforPCA, train_label, name, title):
    print("Drawing %s" % (name))
    model = TSNE(n_components=2)
    source2D = np.ascontiguousarray(model.fit_transform(berforPCA).transpose())
    plt.figure()
    for i in range(train_label.shape[0]):
        plt.scatter(source2D[0][i], source2D[1][i], marker='o', cmap=colors[train_label[i]], s=10, alpha=0.5, label='PCA data')
    plt.title(title)
    plt.show()
    plt.savefig(name)