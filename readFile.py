# https://www.tensorflow.org/datasets/catalog/higgs 
# https://archive-beta.ics.uci.edu/dataset/280/higgs 

f = open("dataset\HIGGS\HIGGS.csv", "r")

if __name__ =='__main__':
    print(f.readline().split(','))
    print(f.readline().split(','))

def readline(n:int):
    tab = []
    for i in range(n):
        tab.append(f.readline().split(','))
    return tab 