import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def gini_impurity(y):
    '''
    Calculating the gini index for determining the impurity of data
    Hàm tính giá trị Gini Index để xác định độ nhiễu của dữ liệu thuộc một class
    '''
    p = y.value_counts() / y.shape[0] # The number of unique values divided by the total number of elements
    gini = 1 - np.sum(p ** 2) # Gini_index formula
    return(gini)


def entropy(y):
    '''
    Calculating the entropy value for determining the impurity of data
    Hàm tính giá trị Entropy để xác định độ nhiễu của dữ liệu thuộc một class
    '''
    a = y.value_counts() / y.shape[0]
    print(a)
    entropy = np.sum(-a * np.log2(a + 1e-9))
    return(entropy)


def variance(y):
    '''
    Calculate the variance avoiding nan.
    Hàm tính phương sai của một ma trận.
    '''
    if(len(y) == 1):
        return 0
    else:
        return y.var()


def information_gain(y, mask, func=entropy):
    '''
    Calculating the Information Gain of a variable given a loss function.
    Using either entrophy function or gini_impurity function.
    Hàm tính giá trị thông tin biết trước giá trị mất mát (loss value) của một vector hoặc một ma trận.
    Có thử sử dụng một trong hai hàm tính toán độ nhiễu thông tin entropy hoặc Gini Index.
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
    Creates all possible combinations to compare the impurity.
    Hàm có chức năng sinh ra tất cả các hoán vị có độ dài từ 1 tới len(a) bao gồm các phần tử trong vector hoặc matrix a.
    Mục đích của hàm này là sinh hoán vị và so sánh độ nhiễu thông tin.
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
    Get the best split based on the maximum information gain value.
    Hàm lấy nút để chia tốt nhất đem lại giá trị thông tin cao nhất.
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
    Dựa vào dữ liệu, hàm sẽ cách chia đem lại giá trị thông tin cao nhất và trả về các giá trị, mảng.
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
    Hàm thực hiện chia dữ liệu.
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
    Hàm dự đoán một mẫu bất kì, có thể có trong tập dữ liệu.
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
    slow down learning process.
    Hàm thực hiện quá trình huấn luyện mô hình máy học sử dụng thuật toán Decision Tree
    '''
    if counter==0:
      types = data.dtypes
      check_columns = types[types == "object"].index
      for column in check_columns:
        var_length = len(data[column].value_counts()) 
        if var_length > max_categories:
          raise ValueError('The variable ' + column + ' has '+ str(var_length) + ' unique values, which is more than the accepted ones: ' +  str(max_categories))

    if max_depth == None:
      depth_cond = True
    else:
      if counter < max_depth:
        depth_cond = True
      else:
        depth_cond = False

    if min_samples_split == None:
      sample_cond = True
    else:
      if data.shape[0] > min_samples_split:
        sample_cond = True
      else:
        sample_cond = False

    if depth_cond & sample_cond:
      var,val,ig,var_type = get_best_split(y, data)
      if ig is not None and ig >= min_information_gain:
        counter += 1
        left,right = make_split(var, val, data,var_type)

        split_type = "<=" if var_type else "in"
        question =   "{} {}  {}".format(var,split_type,val)
        subtree = {question: []}

        yes_answer = train_tree(left,y, target_factor, max_depth,min_samples_split,min_information_gain, counter)
        no_answer = train_tree(right,y, target_factor, max_depth,min_samples_split,min_information_gain, counter)

        if yes_answer == no_answer:
          subtree = yes_answer
        else:
          subtree[question].append(yes_answer)
          subtree[question].append(no_answer)

      else:
        pred = make_prediction(data[y],target_factor)
        return pred

    else:
      pred = make_prediction(data[y],target_factor)
      return pred

    return subtree


def clasificar_datos(observacion, arbol):
    '''
    Hàm xử lý cấu trúc của cây dưới dạng string để đưa về dạng số so sánh và trả về dự đoán.
    '''
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

    if not isinstance(answer, dict):
      return answer
    else:
      residual_tree = answer
      return clasificar_datos(observacion, answer)


def evaluation(X, y, dtree):
    key = [] 
    for i in range(X.shape[0]):
      key.append(clasificar_datos(X.iloc[i], dtree))
    
    return np.mean(key == y)
    

# Nhập vào dữ liệu theo đường dẫn tập dữ liệu đã tải về.
data = pd.read_csv("C:\\Users\\acer\\Downloads\\obese_Dataset\\500_Person_Gender_Height_Weight_Index.csv")
# Tiền xử lý dữ liệu
data['obese'] = (data.Index >= 4).astype('int')
data.drop('Index', axis = 1, inplace = True)
data['Gender'] = np.where(data['Gender'] == 'Male', 1, 0) # Chuyển đổi labels dạng text về dạng số
X = data.iloc[: , : 3]
y = data.iloc[:, 3]

# Thực hiện chia tập dữ liệu thành hai tập trainingset để huấn luyện và validationset để đánh giá với tỉ lệ lần lượt là 8:2
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Huấn luyện mô hình, mô hình trả về có dạng cây nhị phân được viết dưới format String
dtree = train_tree(train_data, 'obese', True, max_depth = 4, min_samples_split = 10, min_information_gain = 1e-5)
X = test_data.iloc[:, :-1]
y = test_data.iloc[:, -1].values

# Đánh giá mô hình
print("Tree after training has a form as: ", dtree)
print("The accuracy of the Decision tree AI model: ", evaluation(X, y, dtree))

test_case = pd.DataFrame({
    'Gender': [0], # 0 là nữ (Female) và 1 là nam (Male)
    'Height': [150],
    'Weight': [110]
})

test_prediction = clasificar_datos(test_case.iloc[0], dtree)
test_prediction = np.where(test_prediction == 1, "The given person is obese", "The given person is normal")
print(test_prediction)



