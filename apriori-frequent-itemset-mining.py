import numpy as np
import csv
import pandas as pd
import itertools as its
import multiprocessing as mp
from collections import defaultdict


def subsets(arr):
    return frozenset(its.chain(*[its.combinations(arr, i + 1) for i, a in enumerate(arr)]))


def joinSet(itemSet, length):
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
    return itemSet, transactionList



def runA(itemSet, transactionList, maxSize, minSupport, queue):

    def returnItemsWithMinSupport(itemSet, transactionList, minSupport):
        _itemSet = set()
        localSet = defaultdict(int)

        for item in itemSet:
            for transaction in transactionList:
                if item.issubset(transaction):
                    localSet[item] += 1

        for item, count in localSet.items():
            support = count

            if support > minSupport:
                _itemSet.add(item)

        return _itemSet


    cat_vec_2 = []
    cat_count_2 = []

    timer = 0

    for k in range(1, 4):
        timer+=1
        print(timer/4*100, "% Completed in Part 2")
        ck = joinSet(itemSet,k)   
        print(k)
        timer2 = 0
        for i in ck:
            timer2+=1
            print(timer2/len(ck)*100, "% completed in nested part 2 with k:", k)
            cat_vec_2.append((";".join(list(i))))
            cat_count_2.append(sum([i<=j for j in transactionList]))
        itemSet = returnItemsWithMinSupport(ck, transactionList, minSupport)
    df2 = pd.DataFrame(columns=['Category','Count'])
    df2['Category'] = cat_vec_2
    df2['Count'] = cat_count_2
    queue.put(df2)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    with open("categories.txt", newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    df = pd.DataFrame(columns=['Category','Count'])
    cat_list = []
    cat_vec = []
    cat_size = []
    cat_count = []
    cat_list_size = []

    ind = 0
    ms_t = 771
    timer = 0

    for i in data:
        timer+=1
        print(timer/len(data)*100, "% Completed in Part 1")
        s = str(i).split(';')
        s[0] = str(s[0])[2:]
        j = len(s)-1
        s[j] = str(s[j])[0:-2]
        s = sorted(s)
        cat_list.append(list(s))
        cat_list_size.append(len(s))
        cat_vec[ind:len(s)] = s
        cat_size[ind:len(s)] = np.repeat(len(s),len(s))
        ind = ind + len(s) + 1



    itemSet, transactionList = getItemSetTransactionList(cat_list)
    maxSize = max(cat_list_size)


    queue = mp.Queue()
    
    
    p1 = mp.Process(target=runA, args=(itemSet, transactionList, maxSize, ms_t, queue))
    p1.start()
    df2 = queue.get()
    p1.join()
    print("mp process")

    min_support2 = df2['Count']>ms_t
    part2 = df2['Count'].astype(str)[min_support2] + ":" + df2['Category'].astype(str)[min_support2]
    np.savetxt("patterns.txt", part2.values, fmt='%-s')
    print("Finished")


