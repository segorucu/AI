import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if (class_index == -1):
        classes = out[:, class_index]
        features = out[:, :class_index]
        return features, classes
    elif (class_index == 0):
        classes = out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """

    # func = lambda feature: feature[2] == 0
    decision_tree_root = DecisionNode(None, None, lambda a: a[0] == 0)
    decision_tree_rootl1 = DecisionNode(None, None, lambda a: a[3] == 0)
    decision_tree_rootl2 = DecisionNode(None, None, lambda a: a[2] == 0)
    decision_tree_root.left = decision_tree_rootl1
    decision_tree_root.right = DecisionNode(None, None, None, 1)
    decision_tree_rootr2 = DecisionNode(None, None, lambda a: a[2] == 0)
    decision_tree_rootr2.left = DecisionNode(None, None, None, 0)
    decision_tree_rootr2.right = DecisionNode(None, None, None, 1)
    decision_tree_rootl1.left = decision_tree_rootl2
    decision_tree_rootl1.right = decision_tree_rootr2
    decision_tree_rootl2.left = DecisionNode(None, None, None, 1)
    decision_tree_rootl2.right = DecisionNode(None, None, None, 0)

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """

    true = np.array(true_labels)
    predict = np.array(classifier_output)
    TP = np.sum(np.logical_and(true == 1, predict == 1))
    FP = np.sum(np.logical_and(true == 0, predict == 1))
    FN = np.sum(np.logical_and(true == 1, predict == 0))
    TN = np.sum(np.logical_and(true == 0, predict == 0))
    mat = [[TP, FN], [FP, TN]]

    return mat


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """
    true = np.array(true_labels)
    predict = np.array(classifier_output)
    return np.sum(np.logical_and(true == 1, predict == 1)) / np.sum(predict == 1)


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """
    true = np.array(true_labels)
    predict = np.array(classifier_output)
    return np.sum(np.logical_and(true == 1, predict == 1)) / np.sum(true == 1)


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """

    true = np.array(true_labels)
    predict = np.array(classifier_output)
    TP = np.sum(np.logical_and(true == 1, predict == 1))
    # FP = np.sum(np.logical_and(true == 0, predict == 1))
    # FN = np.sum(np.logical_and(true == 1, predict == 0))
    TN = np.sum(np.logical_and(true == 0, predict == 0))
    # TODO: finish this.
    return (TP + TN) / len(true_labels)


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    vector = np.array(class_vector)
    p0 = vector[vector == 0].shape[0] / vector.shape[0]
    p1 = vector[vector == 1].shape[0] / vector.shape[0]
    gini_imp = 1.0 - p0 ** 2. - p1 ** 2.

    return gini_imp


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    gini = gini_impurity(previous_classes)
    for clas in current_classes:
        gini -= gini_impurity(clas) * (len(clas) / len(previous_classes))

    return gini


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit
        self.leaf_size = 10

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.
        tree = self.build_tree(features, classes, 0)
        return tree

    def build_tree(self, features, classes, counter):

        counter += 1
        if counter >= self.depth_limit:
            u = np.unique(classes, return_counts=True)
            ind = u[1].argmax()
            val = u[0][ind]
            return DecisionNode(None, None, None, val)
        if features.shape[0] <= self.leaf_size and features.shape[0] != 1:
            u = np.unique(classes, return_counts=True)
            ind = u[1].argmax()
            val = u[0][ind]
            return DecisionNode(None, None, None, val)
        if features.shape[0] == 1 or features.shape[0] <= self.leaf_size:
            # (feature index, separation value, left node, right node)
            return DecisionNode(None, None, None, classes[0])
        if np.max(classes) == np.min(classes):
            return DecisionNode(None, None, None, classes[0])

        gini_max = -1000
        for feature in range(features.shape[1]):
            col = features[:, feature]
            if np.min(col) == np.max(col):
                continue
            Val = np.median(col)
            if (Val == np.max(col) or Val == np.min(col)):
                Val = np.mean(col)
            current_classes = []
            current_classes.append(classes[col <= Val])
            current_classes.append(classes[col > Val])
            try:
                gini = gini_gain(classes, current_classes)
            except:
                print("Houston we have a problem")
            if gini > gini_max:
                gini_max = gini
                SplitVal = Val
                feature_index = feature
        if gini_max == -1000:
            u = np.unique(classes, return_counts=True)
            ind = u[1].argmax()
            val = u[0][ind]
            return DecisionNode(None, None, None, val)

        index = features[:, feature_index] <= SplitVal
        left_tree = self.build_tree(features[index], classes[index], counter)
        index = features[:, feature_index] > SplitVal
        right_tree = self.build_tree(features[index], classes[index], counter)

        decision_tree_root = DecisionNode(None, None, lambda a: (a[feature_index] <= SplitVal, feature_index, SplitVal))
        decision_tree_root.left = left_tree
        decision_tree_root.right = right_tree

        return decision_tree_root

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []
        for feature in features:
            t = self.root
            while (1):
                if t.class_label != None:
                    class_labels.append(t.class_label)
                    break
                elif t.decision_function(feature)[0]:
                    t = t.left
                else:
                    t = t.right

        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    total_size = dataset[0].shape[0]
    size = int(total_size/k)
    shuffler = np.random.permutation(total_size)
    features = dataset[0][shuffler]
    classes = dataset[1][shuffler]
    folds = []
    for i in range(k):
        beg = i * size
        end = (i+1) * size
        features_test = features[beg:end]
        classes_test = classes[beg:end]
        features_training = np.delete(features,list(range(beg,end)),axis=0)
        classes_training = np.delete(classes,list(range(beg,end)),axis=0)
        training_set = (features_training,classes_training)
        testing_set = (features_test,classes_test)
        fold = (training_set,testing_set)
        folds.append(fold)

    return folds


class ChallengeClassifier:
    """Random forest classification."""
    def __init__(self, num_trees = 20, depth_limit = 20, example_subsample_rate = 0.8,
                 attr_subsample_rate = 0.4):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.attributes = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        num_feat = features.shape[1]
        dimy = int(num_feat * self.attr_subsample_rate)
        num_samples = features.shape[0]
        dimx = int(num_samples * self.example_subsample_rate)
        for tree in range(self.num_trees):
            xind = np.random.randint(0, num_samples, dimx)
            yind = np.random.choice(num_feat, dimy, replace=False)
            new_f = features[xind,:]
            new_f = new_f[:,yind]
            new_c = classes[xind]
            Dtree = DecisionTree(depth_limit=self.depth_limit)
            Dtree.fit(new_f,new_c)
            self.trees.append(Dtree)
            self.attributes.append(yind)

        return

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """

        # TODO: finish this.
        classes = []
        output = np.empty((features.shape[0],0))
        for i in range(len(self.trees)):
            DTree = self.trees[i]
            att = self.attributes[i]
            new_f = features[:,att]
            out = DTree.classify(new_f)
            out = np.array(out).reshape((features.shape[0],1))
            output = np.append(output,out,axis=1)
        output = output.astype(int)
        classes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=output)
        return classes


# class ChallengeClassifier:
#     """Challenge Classifier used on Challenge Training Data."""
#
#     def __init__(self, num_trees = 50, depth_limit = 10, example_subsample_rate = 0.5,
#                  attr_subsample_rate = 0.5):
#         """Create a random forest.
#          Args:
#              num_trees (int): fixed number of trees.
#              depth_limit (int): max depth limit of tree.
#              example_subsample_rate (float): percentage of example samples.
#              attr_subsample_rate (float): percentage of attribute samples.
#         """
#
#         self.trees = []
#         self.attributes = []
#         self.num_trees = num_trees
#         self.depth_limit = depth_limit
#         self.example_subsample_rate = example_subsample_rate
#         self.attr_subsample_rate = attr_subsample_rate
#
#     def fit(self, features, classes):
#         """Build a random forest of decision trees using Bootstrap Aggregation.
#             features (m x n): m examples with n features.
#             classes (m x 1): Array of Classes.
#         """
#         dict = {"num_trees": 20, "depth_limit": 20, "example_subsample_rate": 0.8, "attr_subsample_rate": 0.4}
#         myClassifier = RandomForest(**dict)
#
#         return
#
#     def classify(self, features):
#         """Classify a list of features based on the trained random forest.
#         Args:
#             features (m x n): m examples with n features.
#         """
#
#         # TODO: finish this.
#         classes = []
#         output = np.empty((features.shape[0], 0))
#         for i in range(len(self.trees)):
#             DTree = self.trees[i]
#             att = self.attributes[i]
#             new_f = features[:, att]
#             out = DTree.classify(new_f)
#             out = np.array(out).reshape((features.shape[0], 1))
#             output = np.append(output, out, axis=1)
#         classes = output.mean(axis=1)
#         classes = np.round(classes)
#         return classes

class NewDecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit
        self.leaf_size = 10

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.
        tree = self.build_tree(features, classes, 0)
        return tree

    def build_tree(self, features, classes, counter):

        counter += 1
        if counter >= self.depth_limit:
            u = np.unique(classes, return_counts=True)
            ind = u[1].argmax()
            val = u[0][ind]
            return DecisionNode(None, None, None, val)
        if features.shape[0] <= self.leaf_size and features.shape[0] != 1:
            u = np.unique(classes, return_counts=True)
            ind = u[1].argmax()
            val = u[0][ind]
            return DecisionNode(None, None, None, val)
        if features.shape[0] == 1 or features.shape[0] <= self.leaf_size:
            # (feature index, separation value, left node, right node)
            return DecisionNode(None, None, None, classes[0])
        if np.max(classes) == np.min(classes):
            return DecisionNode(None, None, None, classes[0])

        gini_max = -1000
        for feature in range(features.shape[1]):
            col = features[:, feature]
            if np.min(col) == np.max(col):
                continue
            Val = np.median(col)
            if (Val == np.max(col) or Val == np.min(col)):
                Val = np.mean(col)
            current_classes = []
            current_classes.append(classes[col <= Val])
            current_classes.append(classes[col > Val])
            try:
                gini = gini_gain(classes, current_classes)
            except:
                print("Houston we have a problem")
            if gini > gini_max:
                gini_max = gini
                SplitVal = Val
                feature_index = feature
        if gini_max == -1000:
            u = np.unique(classes, return_counts=True)
            ind = u[1].argmax()
            val = u[0][ind]
            return DecisionNode(None, None, None, val)

        index = features[:, feature_index] <= SplitVal
        left_tree = self.build_tree(features[index], classes[index], counter)
        index = features[:, feature_index] > SplitVal
        right_tree = self.build_tree(features[index], classes[index], counter)

        decision_tree_root = DecisionNode(None, None, lambda a: (a[feature_index] <= SplitVal, feature_index, SplitVal))
        decision_tree_root.left = left_tree
        decision_tree_root.right = right_tree

        return decision_tree_root

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []
        for feature in features:
            t = self.root
            while (1):
                if t.class_label != None:
                    class_labels.append(t.class_label)
                    break
                elif t.decision_function(feature)[0]:
                    t = t.left
                else:
                    t = t.right

        return class_labels

class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        # TODO: finish this.
        matrix = np.add(np.multiply(data, data), data)
        return matrix

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this.
        col = np.sum(data[0:100], axis=1)

        return col.max(), col.argmax()

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """
        datan = data[data > 0]
        unique, counts = np.unique(datan, return_counts=True)
        unique_dict = dict(zip(unique, counts))

        # TODO: finish this.
        return unique_dict.items()


def return_your_name():
    # return your name
    # TODO: finish this
    return "sgorucu3"
