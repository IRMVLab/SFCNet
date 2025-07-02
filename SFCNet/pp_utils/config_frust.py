class Config:
    num_classes = 19  # Number of valid classes
    learning_rate = 0.001  # initial learning rate

    total_points = 150000  # 150000/40960

    init_kernel = 3
    out_kernel = 3

    init_H_rm = 64
    init_W_rm = 1800

    conv_dist = [-1., -1., -1., -1.]
    hiden_plane = 128
    aux = True
    block_layers = [3, 4, 6, 3]
    act = 'Hardswish'  # Hardswish SiLU

    trans_conv_kernel = [3, 7, 15]  # 1to0,2to0,3to0

    upper_bound2d_rm = [2.0, 180.]

    lower_bound2d_rm = [-24.8, -180.]

    f_norm = True

    dug = True
    rotate_dug = True
    flip_dug = True
    scale_dug = True
    trans_dug = True
    drop_dug = False

    # initial information setting
    info_list = {'xyz': True, 'range': True, "intensity": True}

    p_loss = False

    noweightce = False

    w_ce = 1.
    w_ls = 1.
