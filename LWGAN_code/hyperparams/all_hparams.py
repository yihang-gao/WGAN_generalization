from utils import registry, hparam

"""
Test for different top_k [1, 10, 20, 30, 40, 50]
"""


@registry.register_hparam('basic_params1')
def basic_params1():
    return hparam.HParam(data_shape=(2,1),
                         buffer_size=2000,
                         batch_size=100,
                         z_shape=2,
                         epochs=50000,
                         test_num=1000,
                         g_depth=2,
                         g_width=30,
                         d_depth=2,
                         d_width=30,
                         lrg=1e-4,
                         lrd=1e-4,
                         beta_1=0.9,
                         beta_2=0.99,
                         bjorck_beta=0.5,
                         bjorck_iter=5,
                         bjorck_order=2,
                         group_size=2,
                         num_critic=5
                         )



@registry.register_hparam('MNist')
def movie_len_1m_params(basic_param_name):
    basic_param_fn = registry.get_hparam(basic_param_name)
    basic_param = basic_param_fn()

    assert isinstance(basic_param, hparam.HParam)

    basic_param.add_params(dataset_name='Synthetic2d'
                           )

    return basic_param
