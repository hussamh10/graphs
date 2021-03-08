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


def add_mentions(fig, mentions, sub, all=True, neg=True, pos=True, neu=False, ratio=True, cumm = False, control=False, means=False, std=None):
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

    color = 'Red'
    scolor = 'rgba(231, 76, 60, 0.2)'
    name = 'Treatment'
    if control:
        scolor = 'rgba(30, 130, 76, 0.2)'
        color = 'limegreen'
        name = 'Control'

    X = list(X)
    
    y = a

    if means:
        s = std['neu']

    if all:
        y = a
        if means:
            s = std['neu']
    if neg:
        y = n
        if means:
            s = std['neg']
    if pos:
        y = p
        if means:
            s = std['pos']
    if neu:
        y = u
        if means:
            s = std['neu']



    fig.add_trace(go.Scatter(y=y,
                        x = X,
                        mode='lines',
                        name=name,
                        line_shape='spline',
                        line =dict(color=color, width=3)))

    if means:
        upper = []
        lower = []
        for i,j in zip(y, s):
            upper.append(i+j)
            lower.append(i-j)

        fig.add_trace(go.Scatter(
            x=X+X[::-1], # x, then x reversed
            y=upper+lower[::-1], # upper, then lower reversed
            fill='toself',
            fillcolor=scolor,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False))
        

        
    return fig

months = ['2015_12', '2016_01', '2016_02', '2016_03', '2016_04', '2016_05', '2016_06', '2016_07', '2016_08', '2016_09', '2016_10', '2016_11', '2016_12', '2017_01', '2017_02', '2017_03', '2017_04', '2017_05', '2017_06', '2017_07', '2017_08', '2017_09', '2017_10', '2017_11', '2017_12', '2018_01', '2018_02', '2018_03', '2018_04', '2018_05', '2018_06', '2018_07', '2018_08', '2018_09', '2018_10', '2018_11', '2018_12', '2019_01', '2019_02', '2019_03', '2019_04', '2019_05', '2019_06', '2019_07', '2019_08']

st.markdown(
        f""" <style> .reportview-container .main .block-container{{ max-width: 5000px; }} </style> """,
        unsafe_allow_html=True,
    )

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


def get_centered_values(pairs, meta, mentions, control_mentions):
    centered_values = dict()
    control_centered_values = dict()

    for sub in pairs:
        centered_values[sub] = dict()
        control_centered_values[pairs[sub]] = dict()
        bm = meta[sub]['ban_date']
        i = months.index(bm) * -1

        for month in mentions[sub]:
            centered_values[sub][i] = mentions[sub][month]
            control_centered_values[pairs[sub]][i] = control_mentions[pairs[sub]][month]
            i += 1
            
    return centered_values, control_centered_values

def means(subs, meta, mentions, control_mentions):
    centered_values, control_centered_values = get_centered_values(subs, meta, mentions, control_mentions)

    mean_values = dict()
    mean_counts = dict()

    c_mean_values = dict()
    c_mean_counts = dict()

    c_std_values = dict()
    std_values = dict()

    for i in range(-23, 23):
        mean_values[i] = {'neg': 0, 'pos': 0, 'neu': 0}
        mean_counts[i] = {'neg': 0, 'pos': 0, 'neu': 0}
        c_mean_values[i] = {'neg': 0, 'pos': 0, 'neu': 0}
        c_mean_counts[i] = {'neg': 0, 'pos': 0, 'neu': 0}

        c_std_values[i] = {'neg': [], 'pos': [], 'neu': []}
        std_values[i] = {'neg': [], 'pos': [], 'neu': []}
        for sub in subs:
            if i in centered_values[sub] and i in control_centered_values[pairs[sub]]:
                for k in mean_values[i]:
                    mean_values[i][k] += centered_values[sub][i][k]
                    std_values[i][k].append(centered_values[sub][i][k])
                    mean_counts[i][k] += 1
                for k in c_mean_values[i]:
                    c_mean_values[i][k] += control_centered_values[pairs[sub]][i][k]
                    c_std_values[i][k].append(control_centered_values[pairs[sub]][i][k])
                    c_mean_counts[i][k] += 1

    for i in mean_values:
        for k in mean_values[i]:
            v = mean_values[i][k]/max(1, mean_counts[i][k])
            v = round(v, 3)
            mean_values[i][k] = v
            std_values[i][k] = np.std(std_values[i][k])


    for i in c_mean_values:
        for k in c_mean_values[i]:
            cv = c_mean_values[i][k]/max(1, c_mean_counts[i][k])
            cv = round(cv, 3)
            c_mean_values[i][k] = cv
            c_std_values[i][k] = np.std(c_std_values[i][k])

    c_std = {'neg': [], 'pos': [], 'neu': []}
    std = {'neg': [], 'pos': [], 'neu': []}
    
    for i in std_values:
        for k in std_values[i]:
            std[k].append(std_values[i][k])

    for i in c_std_values:
        for k in c_std_values[i]:
            c_std[k].append(c_std_values[i][k])


    return mean_values, c_mean_values, std, c_std

(meta, comment_mentions, post_mentions, media_mentions) = load_data()
control_posts = pkl.load(open('control-posts-scored', 'rb'))
control_comments = pkl.load(open('control-comments-scored', 'rb'))

# control_comments = pkl.load(open('control-comments-scored_compound', 'rb'))
# comment_mentions = pkl.load(open('treatment-comments-scored_compound', 'rb'))

# control_posts = pkl.load(open('control-posts-scored_compound', 'rb'))
# post_mentions = pkl.load(open('treatment-posts-scored_compound', 'rb'))

control_media = pkl.load(open('control-selenium-scored', 'rb'))
activity = pkl.load(open('treatment_control_activity', 'rb'))
pairs = pkl.load(open('treatment_control_pairs', 'rb'))
subs = list(pairs.keys())

t_activity = dict()
for s in subs:
    t_activity[s] = activity[s]

t_activity = {k: v for k, v in sorted(t_activity.items(), key=lambda item: item[1])}
subs = list(t_activity.keys())
subs = subs[::-1]

analysis = st.sidebar.selectbox('Show graphs for:', ['Individual subreddits', 'Mean values'])
mention_str = st.sidebar.selectbox('Selection mentions', ['Internal Comments', 'Internal Posts', 'Media'])
mentions_dict = {'Internal Comments': (comment_mentions, control_comments),'Internal Posts': (post_mentions, control_posts), 'Media': (media_mentions, control_media)}


sentiment_str = st.sidebar.selectbox('Mention Sentiment', ['All', 'Negative', 'Positive', 'Neutral'])



all_check = False 
neg_check = False
pos_check = False
neu_check = False

if sentiment_str == 'All':
    all_check = True
if sentiment_str == 'Negative':
    neg_check = True
if sentiment_str == 'Positive':
    pos_check = True
if sentiment_str == 'Neutral':
    neu_check = True

ratio_check = st.sidebar.checkbox('Show percentage', value=False)

cumm_check = st.sidebar.checkbox('Cummulative', value=False)
error_bars = st.sidebar.checkbox('Error Bars', value=False)

fig = go.Figure()
mentions = mentions_dict[mention_str]

if analysis == 'Individual subreddits':
    print('here')
    sub = st.sidebar.selectbox('Select Sub', subs)
    st.sidebar.write(f'{sub} Actvity: {activity[sub]}')
    control = pairs[sub]

    st.sidebar.write(f'Control: {control}')
    st.sidebar.write(f'{control} Actvity: {activity[control]}')

    fig = add_mentions(fig, mentions[0], sub, all=all_check, neg=neg_check, pos=pos_check, neu=neu_check, ratio=ratio_check, cumm=cumm_check, control=False)
    fig = add_mentions(fig, mentions[1], pairs[sub], all=all_check, neg=neg_check, pos=pos_check, neu=neu_check, ratio=ratio_check, cumm=cumm_check, control=True)

    fig = add_regulation(fig, meta[sub]['ban_date'], meta[sub]['intervention'])
    fig = add_layout(fig, ratio_check)

    f'''# {sub}
    Violation: **{meta[sub]['reason']}**  
    ## **{mention_str} and Regulation**  
    '''

else:

    mean_mentions, control_means, err, control_err = means(pairs, meta, mentions[0], mentions[1])
    if error_bars:
        fig = add_mentions(fig, mean_mentions, None, all=all_check, neg=neg_check, pos=pos_check, neu=neu_check, ratio=ratio_check, cumm=cumm_check, control=False, means=True, std=err)
        fig = add_mentions(fig, control_means, None, all=all_check, neg=neg_check, pos=pos_check, neu=neu_check, ratio=ratio_check, cumm=cumm_check, control=True, means=True, std=control_err)
    else:
        fig = add_mentions(fig, mean_mentions, None, all=all_check, neg=neg_check, pos=pos_check, neu=neu_check, ratio=ratio_check, cumm=cumm_check, control=False, means=False, std=err)
        fig = add_mentions(fig, control_means, None, all=all_check, neg=neg_check, pos=pos_check, neu=neu_check, ratio=ratio_check, cumm=cumm_check, control=True, means=False, std=control_err)
    fig = add_regulation(fig, 0)
    fig = add_layout(fig, ratio_check)

    f"""# {type}
    {mention_str} and Regulation
    """

for k in mentions_dict:
    mentions = mentions_dict[k]
    tm, cm = get_centered_values(pairs, meta, mentions[0], mentions[1])
    pkl.dump((tm, cm), open(f'{k}-centered-all', 'wb'))

st.plotly_chart(fig, use_container_width=True)

data = pd.DataFrame.from_dict(pairs, orient='index')
data = data.reindex(index=data.index[::-1])
data.columns = ['Control']
st.dataframe(data)
