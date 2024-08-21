import numpy as np
import pandas as pd
import streamlit as st
import sklearn.model_selection
from sklearn.linear_model import LinearRegression


@st.cache_data
def get_data():
    data = pd.read_csv(r"Bengaluru_House_Data.csv")
    return data


df = get_data()
o = df.copy()
df.drop(['area_type', 'society', 'balcony', 'availability'], axis=1, inplace=True)
df.dropna(inplace=True)
df['bhk'] = df['size'].apply(lambda x: x.split(' ')[0])
df['bhk'] = df['bhk'].astype('int64')


def convert_sqft_to_num(x):
    token = x.split('-')
    if len(token) == 2:
        return float(float(token[0]) + float(token[1])) / 2
    try:
        return float(x)
    except:
        return None


df1 = df.copy()
df1['total_sqft'] = df1['total_sqft'].apply(convert_sqft_to_num)
df2 = df1.copy()
df2['price_per_sqft'] = df2['price'] * 100000 / df2['total_sqft']
df2.location = df2.location.apply(lambda x: x.strip())
l = df2.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats_less_than_10 = l[l <= 10]
df2.location = df2.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
df3 = df2[~(df2.total_sqft / df2.bhk < 300)]


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


df3 = remove_pps_outliers(df3)


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices,
                                            bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')


df4 = remove_bhk_outliers(df3)
df4 = df4[df4.bath < df4.bhk + 2]
df5 = df4.drop(['size', 'price_per_sqft'], axis=1)
dummies = pd.get_dummies(df5.location)
df5 = pd.concat([df5, dummies.drop('other', axis=1)], axis=1)
df5.drop('location', axis=1, inplace=True)
X = df5.drop('price', axis=1)
y = df5['price']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=10)
model = LinearRegression()
model.fit(X_train, y_train)


def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]


# streamlit app
header = st.container()
dataset = st.container()
model_feature_sel = st.container()
output = st.container()

st.markdown('''
        <style>
        .main{
        background-color: #e5eaf5
        }
        </style>''',
            unsafe_allow_html=True
            )
with header:
    st.title('Home Price Predictor')
    st.image('house.jpg')
    st.header('Introduction')
    st.markdown('Introducing our advanced House Price Predictor - your one-stop solution for precise property '
                'appraisal and informed real estate decision-making! This sophisticated tool allows you to track '
                'house prices in Bangalore over specified timeframes, create accurate price projections, and easily '
                'visualise housing patterns.')
    st.markdown('#### Why choose this House Price Predictor:')
    st.markdown('* **Reliable Accuracy:** Using powerful algorithms and robust data, our predictor provides accurate '
                'and reliable house price projections.')
    st.markdown('* **User-Friendly Interface:** Navigate the tool\'s easy interface with ease, making it accessible '
                'to both seasoned real estate professionals and newbies.')
    st.markdown('* **Competitive Edge:** Gain a competitive advantage in property negotiations, investments, '
                'and market analyses by utilising our tool\'s tremendous capabilities.')
with dataset:
    st.header('Dataset')
    st.markdown('This dataset has been taken from "https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price'
                '-data"')
    st.text('Here\'s an overview of the original dataset:')
    st.write(o.head())
    st.markdown('There are only four columns that that are used to predict the price of a house:')
    st.markdown('* location')
    st.markdown('* total square foot')
    st.markdown('* bhk')
    st.markdown('* Number of bathrooms')

with model_feature_sel:
    st.header('Input Features')
    loc = st.selectbox('Select the Location:', options=list(df5.columns[4:]))
    total_sqft = st.slider('Select the total sqft of land:', min_value=min(df5.total_sqft),
                           max_value=max(df5.total_sqft), value=min(df5.total_sqft))
    bhk = st.select_slider('Select BHK:', options=[1, 2, 3, 4, 5])
    bath = st.select_slider('Select Number of bathrooms:', options=[1, 2, 3, 4, 5])

with output:
    st.subheader('Prediction')
    st.markdown('#### The Predicted Price(in Lakhs) is:')
    st.metric(label='**:house:**', value=predict_price(loc, total_sqft, bhk, bath))

    st.warning('The price may change in real-life depending upon the changes in the market!')
