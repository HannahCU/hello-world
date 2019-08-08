import pstats

p = pstats.Stats('output_file')

p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats('cumulative').print_stats(15)
