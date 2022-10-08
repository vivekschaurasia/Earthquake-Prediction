import streamlit 
import joblib
from scipy import stats
from scipy.stats import skew
from scipy.stats import kurtosis
from numpy import cov
from scipy.stats import pearsonr
from scipy.stats import kurtosis , skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


knn_model = joblib.load('C:/Users/vivek/Downloads/Knn.pkl')
RF_model = joblib.load('C:/Users/vivek/Downloads/Forest.pkl')
#XGB_model = joblib.load('C:/Users/vivek/Downloads/RBG.pkl')
#Top 10 important features 
lst = ['2_Std300',
 '3_skew80',
 '2_min80',
 '3_skew300',
 '1_skew10',
 '2_Std80',
 '3_Mean10',
 '2_skew10',
 '2_min300',
 '2_90th precentile10']
#Function 1
def fn1(df):
  #outliers
  dataframe = df[(df["acoustic_data"] >-2.0) & (df["acoustic_data"]<41.0)]

  def feat_transform(x):
    features = {}
    features['Mean'] = x.mean()
    features['Std'] = x.std()
    features['kurtosis']  = x.kurtosis()
    features['skew'] = x.skew()
    features['min'] = x.min()
    features['max'] = x.max()
    features['median'] = x.median()
    features["10th precentile"] = np.percentile(x, 10)
    features["50th precentile"] = np.percentile(x, 50)
    features["90th precentile"] = np.percentile(x, 90)

    for j in [10 , 80 , 300]:
      me = x.rolling(j).mean().dropna()
      features["1_Mean" + str(j)] = me.mean()
      features["1_Std" + str(j)] = me.std()
      features["1_kurtosis" + str(j)] = kurtosis(me)
      features["1_skew" + str(j)] = skew(me)
      features["1_min" + str(j)] = me.min()
      features["1_max" + str(j)] = me.max()
      features["1_median" + str(j)] = me.median()
      features["1_10th precentile"+ str(j)] = np.percentile(me, 10)
      features["1_50th precentile"+ str(j)] = np.percentile(me, 50)
      features["1_90th precentile"+ str(j)] = np.percentile(me, 90)

      st = x.rolling(j).std().dropna()
      features["2_Mean" + str(j)] = st.mean()
      features["2_Std" + str(j)] = st.std()
      features["2_kurtosis" + str(j)] = kurtosis(st)
      features["2_skew" + str(j)] = skew(st)
      features["2_min" + str(j)] = st.min()
      features["2_max" + str(j)] = st.max()
      features["2_median" + str(j)] = st.median()
      features["2_10th precentile"+ str(j)] = np.percentile(st, 10)
      features["2_50th precentile"+ str(j)] = np.percentile(st, 50)
      features["2_90th precentile"+ str(j)] = np.percentile(st, 90)


      skewness = x.rolling(j).skew().dropna()
      features["3_Mean" + str(j)] = skewness.mean()
      features["3_Std" + str(j)] = skewness.std()
      features["3_kurtosis" + str(j)] = kurtosis(skewness)
      features["3_skew" + str(j)] = skew(skewness)
      features["3_min" + str(j)] = skewness.min()
      features["3_max" + str(j)] = skewness.max()
      features["3_median" + str(j)] = skewness.median()
      features["3_10th precentile"+ str(j)] = np.percentile(skewness, 10)
      features["3_50th precentile"+ str(j)] = np.percentile(skewness, 50)
      features["3_90th precentile"+ str(j)] = np.percentile(skewness, 90)
    
    return features 

  feature = feat_transform(dataframe)
  feat = pd.DataFrame(feature)
  new_features = feat.loc[: , lst]
  
  x = new_features.iloc[: , :]

  scalar = joblib.load('C:/Users/vivek/Downloads/scalar.pkl')
  x = scalar.transform(x)
  
   
  knn_model = joblib.load('C:/Users/vivek/Downloads/Knn.pkl')
  RF_model = joblib.load('C:/Users/vivek/Downloads/Forest.pkl')
  #XGB_model = joblib.load('RBG.pkl')
  
  pred = (knn_model.predict(x) + RF_model.predict(x)) / 2
  return float(pred)





def main():
    streamlit.title("Earthquake prediction")
    html_temp = """
    <div style = "background-color:tomato;padding:10px">
    <h2 style = "color:white;text-align:center;">Streamlit earthquake prediction </h2>
    </div>
    """
    streamlit.markdown(html_temp , unsafe_allow_html = True)
    
    file = streamlit.file_uploader("Chose a CSV file" , type = ['csv'])
    
    if file is not None:
        df = pd.read_csv(file)
        result = ""
        if streamlit.button("Predict"):
            result = fn1(df)
            streamlit.success("The output is {}".format(result))
    
if __name__ == "__main__":
    main()