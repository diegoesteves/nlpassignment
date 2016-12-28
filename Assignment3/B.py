from sklearn import svm

import nltk
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import A
from sklearn.feature_extraction import DictVectorizer


# You might change the window size
window_size = 15

# B.1.a,b,c,d
def extract_features(data):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}

    # implement your code here
    for i in data:
        f = dict()
        labels[i[0]] = i[-1]
        token = []
        for s in i[1:-1]:
            nt = nltk.word_tokenize(s)
            token += nt
            if len(nt) == 1:
                word = nt[0]
                idx = token.index(word)

        if idx > len(token) - 1 - window_size:
            for e in token[idx + 1:]:
                f[e] = f.get(e, 0) + 1
        else:
            for e in token[idx + 1:idx + 1 + window_size]:
                f[e] = f.get(e, 0) + 1

        if idx < window_size:
            for e in token[:idx]:
                f[e] = f.get(e, 0) + 1
        else:
            for e in token[idx - window_size:idx]:
                f[e] = f.get(e, 0) + 1

        features[i[0]] = f

    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''



    # implement your code here

    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    selector = SelectKBest(chi2)
    selector.fit(X_train.values(), y_train.values())

    X_train2 = {key: value for key, value in zip(X_train.keys(), selector.transform(X_train.values()))}
    X_test2 = {key: value for key, value in zip(X_test.keys(), selector.transform(X_test.values()))}

    return X_train2, X_test2

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []


    # implement your code here

    svm_clf = svm.LinearSVC()

    X = []
    Y = []
    for i, x in X_train.iteritems():
        X.append(x)
        Y.append(y_train[i])

    svm_clf.fit(X, Y)

    for i, x in X_test.iteritems():
        results.append((i, svm_clf.predict(x)[0]))

    return results

# run part B
def run(train, test, language, answer):
    results = {}

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt])
        test_features, _ = extract_features(test[lexelt])

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)