import numpy as np
# import networkx as nx
from scipy.stats import zscore


class HiCGraphLoader:
    def __init__(self, chromosomes, attributes, K_neighbors=500):
        """
        :param chromosomes: (list)
        :param attributes: (list) names of chosen functional datasets
        :param K_neighbors: only store contacts between a node and its k up-/down- stream nodes
        (shape: [len(chromosomes), 2k + 1])
        """
        self.chromosomes = chromosomes
        self.k_neighbors = K_neighbors

        # Load chromosome sizes
        resolution = 200
        chrom_sizes = [line.strip().split() for line in open('mm10_chrom_sizes.txt')]
        chrom_sizes = {line[0]: int(np.ceil(int(line[1]) / resolution)) for line in chrom_sizes}
        self.chrom_start_ids = [sum([chrom_sizes[c] for c in chromosomes[:i]]) for i in range(len(chromosomes))]
        self.n_nodes = sum([chrom_sizes[c] for c in chromosomes])

        self.adj_info = self.load_adjacency(K_neighbors)
        self.degrees = np.sum(self.adj_info, axis=1)
        self.attributes = self.load_attributes(attributes)

    def load_adjacency(self, K):
        adj = np.zeros((self.n_nodes, 2 * K + 1))
        for chrom in self.chromosomes:
            print('Loading contact map for', chrom)
            line_count = 0
            for line in open(f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/processed/hic_contacts/{chrom}_200bp.txt'):
                line_count += 1
                if line_count % 1000000 == 0:
                    print(' Line:', line_count)
                [p1, p2, w] = line.strip().split()
                p1, p2 = int(p1) // 200, int(p2) // 200
                if abs(p1 - p2) > K:
                    continue
                if p1 == p2:
                    w *= 2
                w = np.log(float(w) + 1)
                adj[p1, K + p2 - p1] += w
                adj[p2, K + p1 - p2] += w
        print('Finish loading contact maps!')
        return adj

    def load_attributes(self, names):
        functional_data = {}
        for chrom in self.chromosomes:
            functional_data[chrom] = None
            for i, k in enumerate(names):
                s = np.loadtxt(
                    f'/nfs/turbo/umms-drjieliu/proj/4dn/data/microC/mESC/processed/{chrom}/{chrom}_200_{k}.txt')
                s = zscore(s)
                print('Loading:', chrom, k, len(s))
                if i == 0:
                    functional_data[chrom] = s
                else:
                    functional_data[chrom] = np.vstack((functional_data[chrom], s))
            functional_data[chrom] = functional_data[chrom].T
            print(functional_data[chrom].shape)
        return functional_data

    def fetch_batch(self, batch_size=100, batch_sampling='degree'):
        while True:
            if batch_sampling == 'degree':
                batch_nodes = np.random.choice(np.arange(self.n_nodes), batch_size, replace=False,
                                               p=np.array(self.degrees) / np.sum(self.degrees))
            elif batch_sampling == 'uniform':
                batch_nodes = np.random.choice(np.arange(self.n_nodes), batch_size)
            else:
                raise ValueError('Wrong Batch Sampling Method!')

            yield [batch_nodes, self.k_neighbors]
