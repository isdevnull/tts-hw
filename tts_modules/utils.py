import io

import matplotlib.pyplot as plt


def plot_image_to_buf(image_data, name=None):
    plt.figure(figsize=(20, 5))
    plt.imshow(image_data)
    plt.title(name)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.clf()
    buf.seek(0)
    return buf
