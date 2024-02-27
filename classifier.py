# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
#from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class Classifier:
    def __init__(self):
        # Initialize multiple models
        self.rf = CustomRandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)
        self.dt = CustomDecisionTreeClassifier(max_depth=100)
        # Create a VotingClassifier ensemble
        self.voting_classifier = VotingClassifier(estimators=[('rf', self.rf), ('dt', self.dt)])

    def reset(self):
        # Reinitialize the models and VotingClassifier
        self.rf = CustomRandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)
        self.dt = CustomDecisionTreeClassifier(max_depth=100)
        # Reset the list of models
        self.voting_classifier = VotingClassifier(estimators=[('rf', self.rf), ('dt', self.dt)])


    def fit(self, data, target):
        # Train the ensemble model with data
        #self.model.fit(data, target)
        self.voting_classifier.fit(data, target)

    def predict(self, data, legal=None):
         # Predict using the VotingClassifier
        predictions = self.voting_classifier.predict([data])
        prediction = predictions[0]  # Assuming [data] was a single instance

        # Map the prediction number to direction string
        number_to_direction = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}
        prediction_str = number_to_direction.get(prediction, 'Stop')
        # Decision logic considering legal moves
        if legal is not None:
            if prediction_str in legal:
                print(f"Predicted and chosen move: {prediction_str}, Legal moves: {legal}")
                return prediction_str
            else:
                chosen = np.random.choice(legal)
                print(f"Predicted move: {prediction_str}, but chosen legal move: {chosen} because it was not legal.")
                return chosen
        else:
            print(f"Predicted move: {prediction_str}, but no legal moves provided.")
            return prediction_str
                
class CustomDecisionTreeClassifier:
    def __init__(self, max_depth=None):
        """
        
        Parameters:
            max_depth (int, optional): The maximum depth of the tree. If None, the tree
            grows until all leaves are pure or until all leaves contain less than
            min_samples_split samples. Defaults to None.
        """
        self.max_depth = max_depth
        self.tree = None  # Root node of the decision tree

    def fit(self, X, y):
        """
        Fits the decision tree classifier on the training data.
        
        Parameters:
            X (list of lists): The training input samples. Each inner list represents
            a vector of features for a single sample.
            y (list): The target values (class labels) for the training samples.
        """
        # Combine features and targets into a single dataset for easier manipulation
        dataset = [list(x) + [y] for x, y in zip(X, y)]
        # Build the decision tree
        self.tree = self.build_tree(dataset, 1)

    def predict(self, X):
        """
        Predicts class labels for the given samples.
        
        Parameters:
            X (list of lists): The input samples to classify.
            
        Returns:
            predictions (list): The list of predicted class labels for each input sample.
        """
        predictions = [self._predict(self.tree, row) for row in X]
        return predictions

    def _gini_impurity(self, groups, classes):
        """
        Calculates the Gini impurity for a split.
        
        Parameters:
            groups (list of lists): The two groups of data split by a feature.
            classes (list): The list of unique class labels in the dataset.
            
        Returns:
            gini (float): The Gini impurity for the groups after the split.
        """
    
        n_instances = float(sum([len(group) for group in groups]))
        gini_scores = np.zeros(len(groups))
        for i, group in enumerate(groups):
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = np.sum([row[-1] == class_val for row in group]) / size
                score += p ** 2
            gini_scores[i] = (1.0 - score) * (size / n_instances)
        return np.sum(gini_scores)
    
    def _test_split(self, index, value, dataset):
        """
        Splits a dataset based on a feature and a feature value.
        
        Parameters:
            index (int): The index of the feature to split on.
            value (float): The value of the feature to split the dataset.
            dataset (list of lists): The dataset to split.
            
        Returns:
            left (list), right (list): Two lists representing the split of the dataset.
        """
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right
        
    

    def _get_split(self, dataset):
        """
        Finds the best feature and value to split the dataset on.
        
        Parameters:
            dataset (list of lists): The dataset to find the best split for.
            
        Returns:
            A dictionary containing the index of the best feature to split on,
            the best value to split on, and the two resulting groups from the split.
        """
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):  # Iterate over all features
            for row in dataset:
                groups = self._test_split(index, row[index], dataset)
                gini = self._gini_impurity(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def _to_terminal(self, group):
        """
        Creates a terminal node value.
        
        Parameters:
            group (list of lists): A group of samples.
            
        Returns:
            The most common output value (class label) in the group.
        """
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def _split(self, node, depth):
        """
        Recursively splits nodes to build the decision tree.
        
        Parameters:
            node (dict): The current node to split.
            depth (int): The current depth in the tree.
        """
        left, right = node['groups']
        del(node['groups'])
        # Check for a no split
        if not left or not right:
            node['left'] = node['right'] = self._to_terminal(left + right)
            return
        # Check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self._to_terminal(left), self._to_terminal(right)
            return
        # Process left child
        if len(left) <= 1:
            node['left'] = self._to_terminal(left)
        else:
            node['left'] = self._get_split(left)
            self._split(node['left'], depth+1)
        # Process right child
        if len(right) <= 1:
            node['right'] = self._to_terminal(right)
        else:
            node['right'] = self._get_split(right)
            self._split(node['right'], depth+1)

    def build_tree(self, train, depth):
        """
        Builds the decision tree.
        
        Parameters:
            train (list of lists): The training dataset.
            depth (int): The initial depth.
            
        Returns:
            The root node of the decision tree.
        """
        root = self._get_split(train)
        self._split(root, depth)
        return root

    def _predict(self, node, row):
        """
        Makes a prediction with the decision tree for a single sample.
        
        Parameters:
            node (dict): The current node of the decision tree.
            row (list): The sample to make a prediction for.
            
        Returns:
            The predicted class label.
        """
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict(node['right'], row)
            else:
                return node['right']

class VotingClassifier:
    def __init__(self, estimators):
        """
        Constructor for the VotingClassifier. This ensemble classifier aggregates predictions from each of the
        models passed in the 'estimators' list, deciding on the final class label based on majority voting.
        
        Parameters:
        - estimators (list of tuples): A list where each tuple contains a model name and the model instance itself.
        """
        self.estimators = estimators  # The list of (name, model) tuples to be used for voting.
        self.fitted_model = []  # This will hold the fitted models after the `fit` method is called.

    def fit(self, X, y):
        """
        Fits each model in the ensemble to the training data. This method iterates over each estimator,
        fits it on the provided data, and stores the fitted model for later use during prediction.
        
        Parameters:
        - X (array-like): The training input samples.
        - y (array-like): The target values (class labels) for the training samples.
        """
        self.fitted_model = []  # Resetting the list to ensure it's empty before fitting new models.
        for name, model in self.estimators:
            model.fit(X, y)  # Fitting each model on the training data.
            self.fitted_model.append((name, model))  # Storing the fitted model.

    def predict(self, X):
        """
        Predicts the class label for each instance in X based on the majority vote from all the fitted models.
        This method collects predictions from each model and then applies a majority voting mechanism to
        decide the final class label for each instance.
        
        Parameters:
        - X (array-like): The input samples to predict.
        
        Returns:
        - final_predictions (list): The predicted class labels for each input sample.
        """
        # Collecting predictions from each fitted model for all instances in X.
        predictions = np.array([model.predict(X) for _, model in self.fitted_model])
        
        # Transposing the prediction array to align predictions for each instance across models.
        predictions = predictions.T
        
        # Determining the majority vote for each instance.
        final_predictions = [Counter(pred).most_common(1)[0][0] for pred in predictions]
        
        return final_predictions

    def predict_with_legal(self, X, legal_moves):
        """
        Extends the `predict` method by ensuring that the predicted class label for each instance is legal.
        This method is particularly useful in scenarios where certain predictions may not be valid or allowed,
        and a legal alternative needs to be selected.
        
        Parameters:
        - X (array-like): The input samples to predict.
        - legal_moves (function): A function that accepts a predicted class label and returns a legal class label.
        
        Returns:
        - legal_predictions (list): The legal predicted class labels for each input sample.
        """
        final_predictions = self.predict(X)  # Getting the initial predictions from the voting classifier.
        
        # Applying the legal_moves function to each prediction to ensure legality.
        legal_predictions = [legal_moves(pred) for pred in final_predictions]
        
        return legal_predictions


# Adding a Sophisticated Classifier: Bagging - Random Forest Classifier

class CustomRandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=7, random_state=42):
        """
        Initializes the custom Random Forest classifier.
        
        Parameters:
        - n_estimators (int): The number of trees in the forest.
        - max_depth (int): The maximum depth of the trees.
        - random_state (int): A seed used by the random number generator for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []  # A list to store the individual trees in the forest.

    def bootstrap_sample(self, X, y):
        """
        Creates a bootstrap sample of the dataset.
        
        Parameters:
        - X (array-like): Feature matrix representing the input data.
        - y (array-like): Target values (labels) for the input data.
        
        Returns:
        - X_sample, y_sample: Bootstrapped samples of the feature matrix and target values.
        """
        indices = np.random.choice(len(X), size=len(X), replace=True)  # Randomly selecting indices with replacement.
        X_sample = np.array(X)[indices]  # Creating the sample feature matrix.
        y_sample = np.array(y)[indices]  # Creating the sample target vector.
        return X_sample, y_sample

    def fit(self, X, y):
        """
        Fits the Random Forest model to the input data.
        
        Parameters:
        - X (array-like): Feature matrix representing the input data.
        - y (array-like): Target values (labels) for the input data.
        """
        self.trees = []  # Resetting/initializing the list of trees.

        def build_tree(seed):
            """
            Inner function to create and train a single decision tree using a bootstrap sample.
            
            Parameters:
            - seed (int): Seed for the random number generator to ensure reproducibility.
            
            Returns:
            - A trained decision tree instance.
            """
            np.random.seed(seed)  # Setting the seed for reproducibility.
            tree = CustomDecisionTreeClassifier(max_depth=self.max_depth)  # Initializing the decision tree classifier.
            X_sample, y_sample = self.bootstrap_sample(X, y)  # Generating a bootstrap sample.
            tree.fit(X_sample, y_sample)  # Fitting the tree to the bootstrap sample.
            return tree

        # Parallelizing the tree building process using ThreadPoolExecutor.
        with ThreadPoolExecutor(max_workers=None) as executor:  # `None` uses as many workers as there are processors.
            futures = [executor.submit(build_tree, seed) for seed in range(self.n_estimators)]  # Submitting tasks to build trees.
            for future in futures:
                self.trees.append(future.result())  # Collecting the trained trees.

    def predict(self, X):
        """
        Predicts the class labels for the given input samples using the trained Random Forest model.
        
        Parameters:
        - X (array-like): Feature matrix representing the input data to predict.
        
        Returns:
        - final_predictions: The predicted class labels for the input samples.
        """
        # Collecting predictions from all trees in the forest.
        predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions_per_instance = predictions.T  # Transposing to align predictions by instance rather than by tree.
        
        # Determining the most common prediction (majority vote) for each instance.
        final_predictions = [Counter(pred).most_common(1)[0][0] for pred in predictions_per_instance]
        return final_predictions
