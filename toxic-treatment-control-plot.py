import pickle as pkl
from random import randint
import streamlit as st

[x, interactions, traits, score] = pkl.load(open('plot-dump', 'rb'))

import streamlit as st
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.write('# Red = Trait, Blue = Interactions')

st.button("Re-run")

pts = 50
x1 = np.arange(pts)
y1 = np.random.random(pts)
y2 = np.random.random(pts)
y3 = (x1/pts)**2

r = randint(0, len(x))
pair = x[r]
c, p = pair
a = traits[c]
i = interactions[pair]

def add_trace(fig, row, col):
    r = randint(0, len(x))
    pair = x[r]
    c, p = pair
    a = traits[c]
    i = interactions[pair]

    fig.add_trace(go.Scatter(y=i, name='interactions', marker_color='blue'),row=row,col=col)
    fig.add_trace(go.Scatter(y=a, name='trait', marker_color='red'),row=row,col=col)

    return fig


fig = make_subplots(rows=5, cols=4)

for r in range(1, 6):
    for c in range(1, 5):
        fig = add_trace(fig, r, c)

fig.update_layout(height=1200, width=2400, title_text="Side By Side Subplots")

g = st.plotly_chart(fig)


