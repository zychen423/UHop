class UHop_PATH():
    def __init__(self, path_pair):
        self.path = {dataset:'../data/PQ/'+path for dataset, path in path_pair}

    def data(self, dataset):
        return self.path[dataset]

    def rela2id(self, dataset):
        return self.path[dataset] + '/rela2id.json'

class Baseline_PATH():
    def __init__(self, path_pair):
        self.path = {dataset:'../data/PQ/baseline/'+path+'/' for dataset, path in path_pair}

    def data(self, dataset):
        return self.path[dataset] + 'MODE_data.txt'

    def rela2id(self, dataset):
        return self.path[dataset] + 'rela2id.json'

    def concat_rela2id(self, dataset):
        return self.path[dataset] + 'concat_rela2id.json'

class WPQ_PATH():
    def __init__(self):
        self.data = {f'wpq{i}':f'../data/PQ/WPQL/{i}/' for i in range(11)}
        self.baseline = {f'wpq{i}':f'../data/PQ/WPQL/baseline/{i}/MODE_data.txt' for i in range(11)}
        self.rela2id = '../data/PQ/WPQL/rela2id.json'
        self.concat_rela2id= '../data/PQ/WPQL/concat_rela2id.json'

def path_finder():
    pair = ([('pq1', 'PQ1'), ('pq2', 'PQ2'), ('pq3', 'PQ3'), ('pql1', 'PQL1'), ('pql2', 'PQL2'), ('pql3', 'PQL3')]
        +   [('pqm'+str(i), 'PQm'+'/'+str(i)+'_'+str(10-i)) for i in range(11)]\
        +   [('pqlm'+str(i), 'PQLm'+'/'+str(i)+'_'+str(10-i)) for i in range(11)])

    return Baseline_PATH(pair), UHop_PATH(pair)
