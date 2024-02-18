# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from sklearn.tree import DecisionTreeClassifier
import numpy as np

from sklearn.ensemble import RandomForestClassifier
import numpy as np

'''
class Classifier:
    def __init__(self):
        # Initialize the Random Forest Classifier with a max_depth
        self.model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)

    def reset(self):
        # Reinitialize the model with the same max_depth
        self.model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)

    def fit(self, data, target):
        # Train the model with data
        self.model.fit(data, target)

    def predict(self, data, legal=None):
        # Predict the move as a number
        prediction = self.model.predict([data])[0]
        
        # Convert prediction number to direction string
        number_to_direction = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}
        prediction_str = number_to_direction.get(prediction, 'Stop')  # Default to 'Stop' if not found
        
        # Print and return the prediction considering legal moves
        if legal is not None:
            if prediction_str in legal:
                print(f"Predicted and chosen move: {prediction_str}, Legal moves: {legal}")
                return prediction_str
            else:
                chosen = np.random.choice(legal)  # Choose a legal move
                print(f"Predicted move: {prediction_str}, but chosen legal move: {chosen} because it was not legal.")
                return chosen
        else:
            print(f"Predicted move: {prediction_str}, but no legal moves provided.")
            return prediction_str
'''

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class Classifier:
    def __init__(self):
        # Initialize multiple models
        self.rf = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
        self.dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        
        # Create a VotingClassifier ensemble
        self.model = VotingClassifier(estimators=[('rf', self.rf), ('dt', self.dt)], voting='hard')

    def reset(self):
        # Reinitialize the models and VotingClassifier
        self.rf = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
        self.dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.model = VotingClassifier(estimators=[('rf', self.rf), ('dt', self.dt)], voting='hard')

    def fit(self, data, target):
        # Train the ensemble model with data
        self.model.fit(data, target)

    def predict(self, data, legal=None):
        # Predict the move as a number
        prediction = self.model.predict([data])[0]
        
        # Map the prediction number to direction string
        number_to_direction = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}
        prediction_str = number_to_direction.get(prediction, 'Stop')
        
        # Decision logic considering legal moves
        if legal is not None:
            if prediction_str in legal:
                print(f"Predicted and chosen move: {prediction_str}, Legal moves: {legal}")
                return prediction_str
            else:
                chosen = np.random.choice(legal)  # Choose a legal move
                print(f"Predicted move: {prediction_str}, but chosen legal move: {chosen} because it was not legal.")
                return chosen
        else:
            print(f"Predicted move: {prediction_str}, but no legal moves provided.")
            return prediction_str

'''

#Random
class RandomAgent(Agent):

    def getAction(self, state):
        # Get the actions we can try, and remove "STOP" if that is one of them.
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # Random choice between the legal options.
        return api.makeMove(random.choice(legal), legal)

# RandomishAgent
#
# A tiny bit more sophisticated. Having picked a direction, keep going
# until that direction is no longer possible. Then make a random
# choice.

class RandomishAgent(Agent):

    # Constructor
    #
    # Create a variable to hold the last action
    def __init__(self):
         self.last = Directions.STOP
    
    def getAction(self, state):
        # Get the actions we can try, and remove "STOP" if that is one of them.
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # If we can repeat the last action, do it. Otherwise make a
        # random choice.
        if self.last in legal:
            return api.makeMove(self.last, legal)
        else:
            pick = random.choice(legal)
            # Since we changed action, record what we did
            self.last = pick
            return api.makeMove(pick, legal)

#Agent 
        
  def registerInitialState(self, state):

        self.data, self.target = loadData('good-moves.txt')
        
        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of integers 0-3 indicating the action
        # taken in that state.            
        self.classifier = Classifier()
        # fit your model to the data
        self.classifier.fit(self.data, self.target)
        
    # Tidy up when Pacman dies
    def final(self, state):
        print("I'm done!")        
        self.classifier.reset()
        
    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)        
        # Get the actions we can try.
        legal = api.legalActions(state)
        
        # predict what action to take
        action = self.convertNumberToMove(self.classifier.predict(features, legal))
        # randomly pick a legal action if the estimated action is illegal
        action = action if action in legal else random.choice(legal)
        
        # getAction has to return a move. We need to pass the set of legal
        # moves to the API so it can do some safety checking.
        return api.makeMove(action, legal)
'''