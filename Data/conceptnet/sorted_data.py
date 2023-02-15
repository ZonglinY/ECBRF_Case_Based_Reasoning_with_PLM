

in_files = ['/home/zy223/CBR/pytorch-transformers-comet/examples/data/conceptnet/train100k_CN.txt', \
            '/home/zy223/CBR/pytorch-transformers-comet/examples/data/conceptnet/dev1_CN.txt', \
            '/home/zy223/CBR/pytorch-transformers-comet/examples/data/conceptnet/dev2_CN.txt', \
            '/home/zy223/CBR/pytorch-transformers-comet/examples/data/conceptnet/test_CN.txt']

out_files = ['/home/zy223/CBR/pytorch-transformers-comet/examples/data/conceptnet/train100k_CN_sorted.txt', \
            '/home/zy223/CBR/pytorch-transformers-comet/examples/data/conceptnet/dev1_CN_sorted.txt', \
            '/home/zy223/CBR/pytorch-transformers-comet/examples/data/conceptnet/dev2_CN_sorted.txt', \
            '/home/zy223/CBR/pytorch-transformers-comet/examples/data/conceptnet/test_CN_sorted.txt']

for id, in_f in enumerate(in_files):
    with open(in_f, 'r') as f:
        lines = f.readlines()
        sorted_lines = sorted(lines)
    with open(out_files[id], 'w') as f:
        f.writelines(sorted_lines)
