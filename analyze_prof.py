import pstats

sts = pstats.Stats('output.prof')
sts.strip_dirs().sort_stats('cumtime').sort_stats(-1).print_stats()