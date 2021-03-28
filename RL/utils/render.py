# For server rendering:
# xvfb-run -s "-screen 0 1400x900x24" jupyter notebook --no-browser --ip=0.0.0.0 --port=8889 &

# For default desktop rendering:
# just use env.render()

import matplotlib.pyplot as plt
from IPython import display


def render(env, title):
    plt.figure(0)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title(title)
    plt.axis('off')
    display.display(plt.gcf())
    display.clear_output(wait=True)