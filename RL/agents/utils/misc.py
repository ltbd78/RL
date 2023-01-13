import numpy as np


def get_total_discounted_rewards(rewards, gamma):
    """
    parameters:
        rewards (r) = [1, 2, 3]
        gamma (γ) = .95

    R(t) = ∑ γ^i * r(i+t) from i=0 to inf

    returns:
        total_discounted_rewards (tdr) = [R(0), R(1), R(2)]
        = [(.95^0)(1)+(.95^1)(2)+(.95^2)(3), (.95^0)(2)+(.95^1)(3), (.95^0)(3)]
        = [5.6075, 4.85, 3]
    """
    tdr = 0
    total_discounted_rewards = []
    for reward in rewards[::-1]:
        tdr = reward + tdr * gamma
        total_discounted_rewards.append(tdr)

    return total_discounted_rewards[::-1]
