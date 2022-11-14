import pandas as pd
import numpy as np

# # make a dummy .tsv file, save it to disk
dummy = pd.DataFrame(np.random.randint(0,10,(200,100)))
save_path = "wikirank-fr-v2.tsv\wikirank-fr-v2.tsv"
df = pd.read_csv(save_path, sep="\t")   # read dummy .tsv file into memory
a = df.values  # access the numpy array containing values



# print(a)
# print("nb ligne = \t" +     str(len(a)))       
# print("nb colonne = \t" +   str(len(a[0])))
colTypes =[type(a[1][i]) for  i in range(len(a[1])) ]
print(colTypes)


# print(df.describe())

# # fourchette de la 1ere colonne 
# ids = [i[0] for i in a]
# print(np.min(ids))
# print(np.max(ids))


# print(df.memory_usage(False,deep=True))


#8
# df.to_csv('chunk',chunksize=int(len(df)*0.1),sep=' ')
for i in range(1,11):
  left = int(len(df)*(0.1*(i-1)))
  right = int(len(df)*(0.1*i))
  print(left,right)
  tab = df[left:right]
  tab.to_csv('chunk'+str(i),sep=' ')


class mychunks:
    def __init__(self):
        self.nbchunks = 10
        self.filename = 'chunk'