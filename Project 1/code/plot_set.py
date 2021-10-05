# Plotting settings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("bmh")
sns.color_palette("hls", 1)

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def color_cycle(num_color):
    """ get color from matplotlib
        color cycle
        use as: color = color_cycle(3) """
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return color[num_color]
