import random
def tri(array):
    step = 0
    for i in range(len(array)):
        minval = array[i]
        for x in range(i,len(array)):
            step +=1
            if(minval > array[x]):
                minval = array[x]
                array[x] = array[i]
                array[i] = minval
    print(step)
    return array

def fusion(arr1,arr2, key=id):
    size = len(arr1)+len(arr2)
    arr = [None] * size
    for i in range(size):
        if  (len(arr1)==0):
            val = arr2.pop(0)
        elif (len(arr2)==0):
            val = arr1.pop(0)
        else:
            if(key(arr1[0])<key(arr2[0])):
                val = arr1.pop(0)
            else : 
                val = arr2.pop(0)
        arr[i] = val
    return arr

def tri_fusion(array,begin=0,end=-1,  key=id):
    """ changer appels recursif de (new array) à (same array,begin,end) pour limiter données alloc """
    if(end==-1):
        end=len(array)
    if(len(array)>=2):
        return fusion(tri_fusion(array[:int(len(array)/2)]) , tri_fusion(array[int(len(array)/2):]),key)
    else:
        return array

tab = [2,5,6,40,20,35,'roar',4,9,900.2,'bwate',200.65]
# tab = tri(tab)
# print(tab)


# tab = tri([])
# tab =  []
# tab = random.sample(range(1000), k= 1000)

# print(len(tab))
# tri(tab)

print(tri_fusion(tab))