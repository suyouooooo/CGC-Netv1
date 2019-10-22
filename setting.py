import os
class CrossValidSetting:
    def __init__(self):
        self.name = 'colon'
        self.batch_size = 16
        self.do_eval = False
        self.sample_time = 1
        self.sample_ratio = 1
        self.root = '/research/dept6/ynzhou/gcnn/data'#'path-to-the-data'#'/research/dept6/ynzhou/gcnn/data'
        self.save_path = '/research/dept6/ynzhou/gcnn/experiment/gcnn-crossval'
        self.log_path = os.path.join(self.save_path,'log' )
        self.result_path = os.path.join(self.save_path, 'result')
        self.dataset = ['shaban']
        self.max_edge_distance = 100
        self.max_num_nodes = 11404 # the maximum number of nodes in one graph
