import os
from os import path
from matplotlib.transforms import Bbox
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text as sk_text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import math
from PIL import Image
import matplotlib as mpl

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

mode = "txt"
# Read the whole text.
if mode == "csv":
    fn = "/gpfs/data/oermannlab/users/lavender/NYUtron/finetuning/finetuning_tasks/readmission/visualization/small/small_both.csv"
    df = pd.read_csv(fn)
    text = df["text"]
    print(df)
    pos_idx = df.index[df["readmitted_in_30_days"] == 1].tolist()
    neg_idx = df.index[df["readmitted_in_30_days"] == 0].tolist()

    # reference: https://towardsdatascience.com/generate-meaningful-word-clouds-in-python-5b85f5668eeb
    # log odd ratio for visualization
    corpus = [
        "".join(df[df["readmitted_in_30_days"] == label].text.tolist())
        for label in [0, 1]
    ]

elif mode == "txt":
    corpus = []
    for label in [0, 1]:
        dir_name = "/gpfs/data/oermannlab/users/lavender/NYUtron/finetuning/finetuning_tasks/readmission/data/deID/results/readmission/"
        with open(dir_name + f"label{label}_all.txt") as f:
            data = f.read()
            corpus.append(data)
else:
    raise Exception(f"mode {mode} not implemented!")

custom_words = []
my_stop_words = sk_text.ENGLISH_STOP_WORDS.union(custom_words)
vec = CountVectorizer(
    stop_words=my_stop_words,
    preprocessor=lambda x: re.sub(r"(\d[\d\.])+", "", x.lower()),
    ngram_range=(1, 1),
)
X = vec.fit_transform(corpus)
X = X.toarray()

bow = pd.DataFrame(X, columns=vec.get_feature_names())
classes = ["non-admitted", "admitted"]
bow.index = classes
print(bow)
t_bow_df = pd.DataFrame()
bow_transformed = bow.apply(lambda x: (x + 1) / bow.loc[x.name].sum() + 1, axis=1)

for label in tqdm(classes):
    others = bow[bow.index != label]
    feat_s = others.sum() + 1
    feat_s_all = np.sum(others.sum())
    tot = feat_s / feat_s_all
    row = bow_transformed.loc[label] / tot
    results = row.apply(lambda x: math.log(x, 2))
    t_bow_df = pd.concat([t_bow_df, pd.DataFrame([results], index=[label])])
    print(t_bow_df)

# cut out background: https://pixlr.com/remove-background/
# recolor to black: https://pinetools.com/colorize-image
mask_d = {"non-admitted": "negative.jpg", "admitted": "positive.jpg"}
cmap = mpl.cm.Blues(np.linspace(0, 1, 20))
cmap = mpl.colors.ListedColormap(cmap[-10:, :-1])
for label in classes:
    print(label)
    res = t_bow_df.loc[label].sort_values(ascending=False)[:100]
    print(res[:10])
    print("==========")
    td = {k: v for k, v in sorted(res.items(), reverse=True, key=lambda item: item[1])}
    img_mask = np.array(Image.open(f"wordclouds/{mask_d[label]}"))
    img_mask = np.where(img_mask > 3, 255, img_mask)
    wordcloud = WordCloud(
        width=800,
        height=400,
        min_word_length=3,
        background_color="white",
        contour_color="purple",
        mask=img_mask,
        colormap=cmap,
        contour_width=4,
    ).generate_from_frequencies(td)
    fig = plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(f"wordclouds/test_{label}.png", dpi=400)
