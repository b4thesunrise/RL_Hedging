from gym.envs.registration import register

register(
    id='DHENV-v0',
    entry_point='DHENV.envs:GBM_simple',
)

register(
    id='DHENV-v1',
    entry_point='DHENV.envs:GBM_simple_PL_cdf',
)