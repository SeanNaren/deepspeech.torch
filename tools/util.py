import os
from optparse import OptionParser
from itertools import izip

parser = OptionParser()
parser.add_option('--mode', dest='MODE', type='string', help='')
parser.add_option('--file', dest='FILE', type='string', help='')
parser.add_option('--file2', dest='FILE2', type='string', help='')
(options, args) = parser.parse_args()

def make_index_file(trans_file, path_file):
    f = open('index.txt', 'w')
    
    with open(trans_file) as f2, open(path_file) as f1:
        for l1,l2 in izip(f1, f2):
            l1 = l1.strip()
            l2 = l2.strip()
            print >> f, '%s@%s@' % ('/data1/yuanyang/torch_projects/data/an4/wav/'+l1+'.wav',l2) #TODO hard code

    f.close()
    
    
    
    

if __name__ == '__main__':
    if options.MODE == 'make_index_file':
        make_index_file(options.FILE, options.FILE2)
