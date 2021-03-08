from os import supports_bytes_environ
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle as pkl
import streamlit as st
import plotly.graph_objects as go


def get_ban_mentions(sub, mentions):
    date = meta[sub]['ban_date']
    ban_mentions = {'neg':0, 'pos':0, 'neu':0}

    for month in mentions[sub]:
        if month < date:
            for t in mentions[sub][month]:
                ban_mentions[t] += mentions[sub][month][t]
    return ban_mentions

def get_dist(subs, mentions, sentiment):
    max_neg=0
    ban_mentions = dict()
    for sub in subs:
        ban_mentions[sub] = get_ban_mentions(sub, mentions)
        if ban_mentions[sub][sentiment] > max_neg:
            max_neg = ban_mentions[sub][sentiment]
            
    ban_count = dict()
    ban_count[0] = 0
    for i in range(1, max_neg+11, 10):
        ban_count[i] = 0
        for sub in subs:
            if ban_mentions[sub][sentiment] < i:
                ban_count[i] += 1
        ban_count[i] = ban_count[i]/len(subs)
        
    return ban_count

def get_pdf_plot(mentions, sentiment, subs):
    sub_mentions = dict()
    for sub in subs:
        sub_mentions[sub] = mentions[sub]
    d = get_dist(subs, sub_mentions, sentiment)
    X, Y = list(d.keys()), list(d.values())
    return X, Y

def load_data():
    return pkl.load(open('new-dump', 'rb'))

def add_regulation(fig, info, type='banned'):
    if type == 'quarantined':
        color = 'rgba(255,193,7,0.1)'
        color2 = 'rgba(255,193,7,.6)'
    else:
        color = 'rgba(255,87,34,0.1)'
        color2 = 'rgba(255,87,34,.6)'
        
    fig.add_vline(info, line=dict(color=color, width=30))
    fig.add_vline(info, line=dict(color=color2, width=3))
    
    return fig


def add_mentions(fig, mentions, sub, all=True, neg=True, pos=True, neu=False, ratio=True, cumm = False):
    if sub == None:
        df = pd.DataFrame(mentions).T
    else:
        df = pd.DataFrame(mentions[sub]).T
    
    if ratio:
        a = (df['pos'] + df['neg'] + df['neu'])/(df['pos'] + df['neg'] + df['neu']) * 100
        n = df['neg']/(df['pos'] + df['neg'] + df['neu']) * 100
        p = df['pos']/(df['pos'] + df['neg'] + df['neu']) * 100
        u = df['neu']/(df['pos'] + df['neg'] + df['neu']) * 100
    else:
        a = (df['pos'] + df['neg'] + df['neu'])
        n = df['neg']
        p = df['pos']
        u = df['neu']


    X = df.index

    if cumm:
        a = np.cumsum(a)
        n = np.cumsum(n)
        p = np.cumsum(p)
        u = np.cumsum(u)

    if all:
        fig.add_trace(go.Scatter(y=a,
                            x = X,
                            mode='lines',
                            line =dict(color='Blue', width=2),
                            name='All Mentions'))
    
    if neg:
        fig.add_trace(go.Scatter(y=n,
                            x = X,
                            mode='lines',
                            line =dict(color='Red', width=2),
                            name='Negative Mentions'))
    
    if pos:
        fig.add_trace(go.Scatter(y=p,
                            x = X,
                            mode='lines',
                            line =dict(color='Green', width=2),
                            name='Positive Mentions'))
        
        
    if neu:
        fig.add_trace(go.Scatter(y=u,
                            x = X,
                            mode='lines',
                            line =dict(color='Gray', width=2),
                            name='Neutral Mentions'))
        
    return fig

months = ['2015_12', '2016_01', '2016_02', '2016_03', '2016_04', '2016_05', '2016_06', '2016_07', '2016_08', '2016_09', '2016_10', '2016_11', '2016_12', '2017_01', '2017_02', '2017_03', '2017_04', '2017_05', '2017_06', '2017_07', '2017_08', '2017_09', '2017_10', '2017_11', '2017_12', '2018_01', '2018_02', '2018_03', '2018_04', '2018_05', '2018_06', '2018_07', '2018_08', '2018_09', '2018_10', '2018_11', '2018_12', '2019_01', '2019_02', '2019_03', '2019_04', '2019_05', '2019_06', '2019_07', '2019_08']

st.markdown(
        f""" <style> .reportview-container .main .block-container{{ max-width: 5000px; }} </style> """,
        unsafe_allow_html=True,
    )


def get_sub_types(meta, subs):
    types = dict()
    types['All'] = []
    for sub in subs:
        types['All'].append(sub)
        type = meta[sub]['reason']
        if type not in types:
            types[type] = []
        types[type].append(sub)

    for type in types:
        types[type] = sorted(types[type])

    return types

def add_layout(fig, ratio_check):
    if ratio_check:
        ratio_str = '%'
    else:
        ratio_str = 'count'

    fig.update_layout(
        width=2000,
        height=600,
        yaxis=dict(
            title_text=f"Mentions {ratio_str}",
            titlefont=dict(size=20),
        ),
        xaxis=dict(
            title_text="Months",
            titlefont=dict(size=20),
        )
    )

    return fig


def get_centered_values(subs, meta, mentions):
    centered_values = dict()

    for sub in subs:
        centered_values[sub] = dict()
        bm = meta[sub]['ban_date']
        i = months.index(bm) * -1

        for month in mentions[sub]:
            centered_values[sub][i] = mentions[sub][month]
            i += 1
            
    return centered_values

def means(subs, meta, mentions):
    centered_values = get_centered_values(subs, meta, mentions)
    
    mean_values = dict()
    mean_counts = dict()

    for i in range(-23, 23):
        mean_values[i] = {'neg': 0, 'pos': 0, 'neu': 0}
        mean_counts[i] = {'neg': 0, 'pos': 0, 'neu': 0}
        for sub in subs:
            if i in centered_values[sub]:
                for k in mean_values[i]:
                    mean_values[i][k] += centered_values[sub][i][k]
                    mean_counts[i][k] += 1

    for i in mean_values:
        for k in mean_values[i]:
            v = mean_values[i][k]/max(1, mean_counts[i][k])
            v = round(v, 3)
            mean_values[i][k] = v

    return mean_values

def get_box_values(subs, meta, mentions):
    r = list(range(-23, 23))
    cv = get_centered_values(subs, meta, mentions)

    neg_sum = []
    pos_sum = []
    neu_sum = []
    all_sum = []

    for sub in subs:
        for i in cv[sub]:
            if i in r:
                v = cv[sub][i]
                neg_sum.append(v['neg'])
                pos_sum.append(v['pos'])
                neu_sum.append(v['neu'])
                all_sum.append(v['neg'] + v['pos'] + v['neu'])

    box_values = {'neg': neg_sum, 'pos': pos_sum, 'neu': neu_sum, 'all': all_sum}
    return box_values

(meta, comment_mentions, post_mentions, media_mentions) = load_data()
subs = list(meta.keys())

types = get_sub_types(meta, subs)
keys = sorted(list(types))

analysis = st.sidebar.selectbox('Show graphs for:', ['Individual subreddits', 'Mean values'])

mention_str = st.sidebar.selectbox('Selection mentions', ['Internal Comments', 'Internal Posts', 'Media'])

mentions_dict = {'Internal Comments': comment_mentions,'Internal Posts': post_mentions, 'Media': media_mentions}

all_check = st.sidebar.checkbox('All mentions', value=True)
neg_check = st.sidebar.checkbox('Negative mentions')
pos_check = st.sidebar.checkbox('Positive mentions')
neu_check = st.sidebar.checkbox('Neurtal mentions')
ratio_check = st.sidebar.checkbox('Show percentage', value=False)

cumm_check = st.sidebar.checkbox('Cummulative', value=False)

type = st.sidebar.selectbox('Select Violation', keys)

fig = go.Figure()
mentions = mentions_dict[mention_str]

if analysis == 'Individual subreddits':
    print('here')
    sub = st.sidebar.selectbox('Select Sub', types[type])

    fig = add_mentions(fig, mentions, sub, all=all_check, neg=neg_check, pos=pos_check, neu=neu_check, ratio=ratio_check, cumm=cumm_check)
    fig = add_regulation(fig, meta[sub]['ban_date'], meta[sub]['intervention'])
    fig = add_layout(fig, ratio_check)

    f'''# {sub}
    Violation: **{meta[sub]['reason']}**  
    ## **{mention_str} and Regulation**  
    '''

else:

    mean_mentions = means(types[type], meta, mentions)
    fig = add_mentions(fig, mean_mentions, None, all=all_check, neg=neg_check, pos=pos_check, neu=neu_check, ratio=ratio_check, cumm=cumm_check)
    fig = add_regulation(fig, 0)
    fig = add_layout(fig, ratio_check)

    f"""# {type}
    {mention_str} and Regulation
    """


st.plotly_chart(fig, use_container_width=True)

import plotly.graph_objects as go
fig = go.Figure()


col1, col2 = st.beta_columns((1, 4))

types_selected = []

for t in types:
    if col1.checkbox(t):
        types_selected.append(t)

sentiment = 'neg'
if neg_check:
    sentiment = 'neg'
if pos_check:
    sentiment = 'pos'
if neu_check:
    sentiment = 'neu'


for t in types_selected:

    X, Y = get_pdf_plot(mentions, sentiment, types[t])
    fig.add_trace(go.Scatter(y=Y,
                        x = X,
                        mode='lines',
                        name=f'{mention_str} - {sentiment} - {t}'))

fig.update_layout(
        width=800,
        height=800,
        yaxis=dict(
            title_text=f"Probability of Banning",
            titlefont=dict(size=20),
        ),
        xaxis=dict(
            title_text="Number of Mentions",
            titlefont=dict(size=20),
        )
    )

col2.plotly_chart(fig, use_container_width=True)


# if comment_check:
#     mentions = comment_mentions
# if post_check:
#     mentions = post_mentions
# if media_check:
#     mentions = media_mentions

# box_values = get_box_values(subs, meta, mentions)

# import plotly.express as px
# df = px.data.tips()

# dfy = []
# dfx = []

# if neg_check:
#     dfy += box_values['neg']
#     dfx += ['Negative'] * len(box_values['all'])
# if pos_check:
#     dfy += box_values['pos']
#     dfx += ['Positive'] * len(box_values['all'])
# if neu_check:
#     dfy += box_values['neu']
#     dfx += ['Neutral'] * len(box_values['all'])
# if all_check:
#     dfy += box_values['all']
#     dfx += ['All'] * len(box_values['all'])

# df = pd.DataFrame()
# df['X'] = dfx
# df['Y'] = dfy

# df

# fig = px.box(df, x="X", y="Y")
# st.plotly_chart(fig, use_container_width=True)
