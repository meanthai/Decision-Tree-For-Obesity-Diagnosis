import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split

def gini_impurity(y):
    '''
    Calculating the gini index for determining the impurity
    '''
    p = y.value_counts() / y.shape[0] # The number of unique values divided by the total number of elements
    gini = 1 - np.sum(p ** 2) # Gini_index formula
    return(gini)


def entropy(y):
    '''
    Calculating the entrophy value for determining the impurity
    '''
    a = y.value_counts() / y.shape[0]
    print(a)
    entropy = np.sum(-a * np.log2(a + 1e-9))
    return(entropy)


def variance(y):
    '''
    Calculate the variance avoiding nan.
    '''
    if(len(y) == 1):
        return 0
    else:
        return y.var()


def information_gain(y, mask, func=entropy):
    '''
    Calculating the Information Gain of a variable given a loss function.
    Using either entrophy function or gini_impurity function
    '''
    a = sum(mask)
    b = mask.shape[0] - a
    if(a == 0 or b ==0):
        ig = 0
    
    else:
        if y.dtypes != 'O': # For numeric data
            ig = variance(y) - (a / (a + b) * variance(y[mask])) - (b / (a + b) * variance(y[-mask]))
        else: # For categorical data
            ig = func(y) - a / (a + b) * func(y[mask]) - b / (a + b) * func(y[-mask])
    
    return ig


def categorical_options(a):
    '''
    Creates all possible combinations to compare the impurity
    '''
    a = a.unique()

    opciones = []
    for L in range(1, len(a)+1):
        for subset in itertools.combinations(a, L):
            subset = list(subset)
            opciones.append(subset)

    return opciones[:-1]


def max_information_gain_split(x, y, func=entropy):
    '''
    Get the best split based on the maximum information gain value
    '''
    split_value = []
    ig = [] 

    # Converting word-related features into numeric variables
    numeric_variable = 1 if x.dtypes != 'O' else 0
    if numeric_variable:
        options = x.sort_values().unique()[1:]
    else: 
        options = categorical_options(x)

    # Calculate ig for all values
    for val in options:
        mask = x < val if numeric_variable else x.isin(val)
        val_ig = information_gain(y, mask, func)
        ig.append(val_ig)
        split_value.append(val)

    # Check if there are more than 1 results if not, return False
    if len(ig) == 0:
        return(None, None , None, False)

    else:
    # Get results with highest IG
        best_ig = max(ig)
        best_ig_index = ig.index(best_ig)
        best_split = split_value[best_ig_index]
        return(best_ig,best_split,numeric_variable, True)


def get_best_split(y, data):
    '''
    Given a data, select the best split and return the variable, the value, the variable type and the information gain.
    '''
    masks = data.drop(y, axis= 1).apply(max_information_gain_split, y = data[y])
    if sum(masks.loc[3,:]) == 0:
        return(None, None, None, None)

    else:
        # Get only masks that can be splitted
        masks = masks.loc[:,masks.loc[3,:]]

        # Get the results for split with highest IG
        split_variable = masks.iloc[0].astype(np.float32).idxmax()
        split_value = masks[split_variable][1] 
        split_ig = masks[split_variable][0]
        split_numeric = masks[split_variable][2]

        return(split_variable, split_value, split_ig, split_numeric)


def make_split(variable, value, data, is_numeric):
    '''
    Given a data and a split conditions, do the split.
    '''
    if is_numeric:
        branch_1 = data[data[variable] < value]
        branch_2 = data[data[variable] >= value]

    else:
        branch_1 = data[data[variable].isin(value)]
        branch_2 = data[not(data[variable].isin(value))]

    return(branch_1, branch_2)

def make_prediction(data, target_factor):
    '''
    Given the target variable, make a prediction.
    '''
    # Make predictions
    if target_factor:
        pred = data.value_counts().idxmax()
    else:
        pred = data.mean()

    return pred

def train_tree(data, y, target_factor, max_depth = None,min_samples_split = None, min_information_gain = 1e-20, counter=0, max_categories = 20):
    '''
    Trains a Decission Tree
    slow down learning process. R
    '''
    # Check that max_categories is fulfilled
    if counter==0:
      types = data.dtypes
      check_columns = types[types == "object"].index
      for column in check_columns:
        var_length = len(data[column].value_counts()) 
        if var_length > max_categories:
          raise ValueError('The variable ' + column + ' has '+ str(var_length) + ' unique values, which is more than the accepted ones: ' +  str(max_categories))

    # Check for depth conditions
    if max_depth == None:
      depth_cond = True
    else:
      if counter < max_depth:
        depth_cond = True
      else:
        depth_cond = False

    # Check for sample conditions
    if min_samples_split == None:
      sample_cond = True
    else:
      if data.shape[0] > min_samples_split:
        sample_cond = True
      else:
        sample_cond = False

    # Check for ig condition
    if depth_cond & sample_cond:
      var,val,ig,var_type = get_best_split(y, data)
      # If ig condition is fulfilled, make split 
      if ig is not None and ig >= min_information_gain:
        counter += 1
        left,right = make_split(var, val, data,var_type)

        # Instantiate sub-tree
        split_type = "<=" if var_type else "in"
        question =   "{} {}  {}".format(var,split_type,val)
        # question = "\n" + counter*" " + "|->" + var + " " + split_type + " " + str(val) 
        subtree = {question: []}

        # Find answers (recursion)
        yes_answer = train_tree(left,y, target_factor, max_depth,min_samples_split,min_information_gain, counter)
        no_answer = train_tree(right,y, target_factor, max_depth,min_samples_split,min_information_gain, counter)

        if yes_answer == no_answer:
          subtree = yes_answer
        else:
          subtree[question].append(yes_answer)
          subtree[question].append(no_answer)

      # If it doesn't match IG condition, make prediction
      else:
        pred = make_prediction(data[y],target_factor)
        return pred

    # Drop dataset if doesn't match depth or sample conditions
    else:
      pred = make_prediction(data[y],target_factor)
      return pred

    return subtree


# Binary Classification Function whether a person checked is obese or not
def clasificar_datos(observacion, arbol):
    question = list(arbol.keys())[0] 

    if question.split()[1] == '<=':
      if observacion[question.split()[0]] <= float(question.split()[2]):
        answer = arbol[question][0]
      else:
        answer = arbol[question][1]

    else:
      if observacion[question.split()[0]] in (question.split()[2]):
        answer = arbol[question][0]
      else:
        answer = arbol[question][1]

    # If the answer is not a dictionary
    if not isinstance(answer, dict):
      return answer
    else:
      residual_tree = answer
      return clasificar_datos(observacion, answer)


def evaluation(X, y, dtree):
    key = [] # Store the predicted labels
    for i in range(X.shape[0]):
      key.append(clasificar_datos(X.iloc[i], dtree))
    
    return np.mean(key == y)
    

data = pd.read_csv("C:\\Users\\acer\\Downloads\\obese_Dataset\\500_Person_Gender_Height_Weight_Index.csv")
# Preprocessing dataset
data['obese'] = (data.Index >= 4).astype('int')
data.drop('Index', axis = 1, inplace = True)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

dtree = train_tree(train_data, 'obese', True, max_depth = 4, min_samples_split = 10, min_information_gain = 1e-5)
X = test_data.iloc[:, :-1]
y = test_data.iloc[:, -1].values

print("The accuracy of the Decision tree AI model: ", evaluation(X, y, dtree))


