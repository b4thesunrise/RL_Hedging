from gym.envs.registration import register

register(
    id='DHENV-v0',
    entry_point='DHENV.envs:GBM_simple',
)

register(
    id='DHENV-cashflow',
    entry_point='DHENV.envs:GBM_cashflow',
)