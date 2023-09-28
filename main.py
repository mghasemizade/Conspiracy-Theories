import pandas as pd
from imports import *
from classifier import predict
from clustering import bayesian_search
from labeling_1 import extract_labels
from labeling_2 import label_clusters_with_dict, ft
from ner import entity_count

def main(csv_path, text_column, threshold):
    df = pd.read_csv(csv_path, encoding='utf-8')
    df['label'] = df[text_column].apply(lambda text: predict(text, threshold)) #predict the text column in the dataset with our classifier
    df1 = df[df['label'] == 1] #keeping only the ones prediced as CT

    sentences = list(df1['text'])
    model = SentenceTransformer('sentence-transformers/stsb-roberta-base-v2') #using roberta sentence encoder to generate vectors for each word
    embeddings = model.encode(sentences)

    #define a grid space for bayesian hyperparameters search, based on the smallest cost
    hspace = {
        'n_neighbors': hp.choice('n_neighbors', range(3, 20)),
        'n_components': hp.choice('n_components', range(3, 20)),
        'min_cluster_size': hp.choice('min_cluster_size', range(2, 20)),
        'random_state': 42

    }
    label_lower = 30
    label_upper = 100
    max_evals = 100
    #finding the best cluster model
    best_params, best_cluster, trial = bayesian_search(embeddings,
                                                       space=hspace,
                                                       label_lower=label_lower,
                                                       label_upper=label_upper,
                                                       max_evals=max_evals)
    #saving the cluster numbers in df1
    df1.loc[:, 'Cluster'] = best_cluster.labels_

    #generating labels with the first method, and saving it in a label1 column
    clusters = {label: [] for label in np.unique(best_cluster.labels_)}
    for doc, label in zip(sentences, best_cluster.labels_):
        clusters[label].append(doc)

    cluster_labels = {label: extract_labels(docs) for label, docs in clusters.items()}

    df1['label1'] = df1['Cluster'].map(cluster_labels)
    df1 = label_clusters_with_dict(df1, ft, 'Cluster', 'text', 'label2')
    entity_count(clusters)
    df = pd.merge(df, df1[['Cluster', 'label1', 'label2']], how='left', left_index=True, right_index=True)
    df.to_csv('output.csv', index=False)

if __name__ == '__main__':
    main('small_df.csv', 'text', 0.5)
