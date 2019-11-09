import traceback

try:
    from tf_nndistance import nn_distance
    from tf_approxmatch import approx_match, match_cost
except Exception as e:
    traceback.print_exc()

    print('External Losses (Chamfer-EMD) were not loaded.')
