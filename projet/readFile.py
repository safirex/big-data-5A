import pandas as pd

white = open("dataset/winequality-white.csv", "r")
red = open("dataset/winequality-red.csv", "r")

# get rid of csv header
# white.readline()
# red.readline()

def get_column_labels():
    column_names= ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','ph','sulphates','alcohol','quality']
    return column_names

def read_red(n:int):
    tab = []
    for i in range(n+1):
        tab.append(red.readline().replace('\n','').split(';'))
    return pd.DataFrame(tab[1:],columns=get_column_labels()) 

def read_white(n:int):
    tab = []
    for i in range(n+1):
        tab.append(white.readline().replace('\n','').split(';'))
    return pd.DataFrame(tab[1:],columns=get_column_labels()) 
