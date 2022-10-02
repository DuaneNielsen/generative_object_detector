import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image, ImageDraw, ImageFont
import os


class Prediction:
    def __init__(self, line, labels=None):
        splt = line.rstrip().split(' ')
        self.line = line
        self.image_fn, self.preds = splt[0], np.array([float(n) for n in splt[1].split(',')])
        self.diff = (self.preds[0] - self.preds[1]) ** 2
        self.label = np.argmax(self.preds)
        self.label_txt = labels[self.label] if labels else str(self.label)

    def PIL_image(self, class_caption=False, labels=None):
        image = Image.open(self.image_fn)
        if class_caption:
            caption = f'{self.label_txt}'
            d = ImageDraw.Draw(image)
            font = ImageFont.truetype('resources/DejaVuSans.ttf', size=50)
            d.text((10, 400), caption, fill='white', font=font,
                   stroke_width=2, stroke_fill='black')
        return image

    def __str__(self):
        return self.line

    def __repr__(self):
        return f'Pred < {self.image_fn} {str(self.preds)} {str(self.diff)} >'


if __name__ == '__main__':

    preds_fn = 'preds.txt'
    labels_fn = 'labels.txt'

    labels = []

    with open(labels_fn) as f:
        for line in f:
            labels.append(line)

    preds = []

    with open(preds_fn, 'r') as f:
        for line in f:
            preds.append(Prediction(line, labels))

    preds = sorted(preds, key=lambda x: x.diff)

    for pred in preds:
        print(pred)

    freqs = np.arange(2, 20, 3)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    l = ax.imshow(preds[0].PIL_image(class_caption=True))


    class Index:
        ind = 0

        def next(self, event):
            if self.ind < len(preds):
                self.ind += 1
            l.set_data(preds[self.ind].PIL_image(class_caption=True))
            plt.draw()

        def prev(self, event):
            if self.ind > 0:
                self.ind -= 1
            l.set_data(preds[self.ind].PIL_image(class_caption=True))
            plt.draw()

        def delete(self, event):
            os.remove(preds[self.ind].image_fn)
            del preds[self.ind]

            with open(preds_fn, "w") as f:
                for pred in preds:
                    f.write(pred.line)

            l.set_data(preds[self.ind].PIL_image(class_caption=True))
            plt.draw()


    callback = Index()
    axprev = fig.add_axes([0.59, 0.05, 0.1, 0.075])
    axdel = fig.add_axes([0.7, 0.05, 0.1, 0.075])
    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    bdel = Button(axdel, 'Delete')
    bdel.on_clicked(callback.delete)
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    plt.show()
