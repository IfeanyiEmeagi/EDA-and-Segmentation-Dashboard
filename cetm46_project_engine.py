#Subject: CETM46 - A Domain Specific Data Science Product Development Project.
#Title: EDA and Customer Segmentation in Banking industry.
#Objective - Identify customers that are likely to purchase term deposit using EDA 
#and cluster customers with similar features in a group.
#Data Source: Kaggle Dataset - Customer Segment

#Import the required library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


import warnings
warnings.filterwarnings("ignore")


def cluster(data, k):
  """The function cluster the scaled data into groups by generating labels for each group.
  Parameters
  ==========
  Input: data: Dataframe, k: int

  Return
  ======
  Output: Label: ndarray
  """
  #Instantiate the model
  model = KMeans(n_clusters = k, random_state = 42)

  #fit the model
  model.fit(data)

  #Extract the group labels
  labels = model.labels_

  #Extract the centroid of the labels
  centers = model.cluster_centers_

  return labels, centers

def quantile(data, column_name):
  q1 = data[column_name].quantile(0.25)
  q3 = data[column_name].quantile(0.75)

  iqr = q3 - q1

  upper_bound = q3 + 1.5 * iqr
  lower_bound = q1 - 1.5 * iqr

  data[column_name] = np.where(data[column_name] < lower_bound, lower_bound, data[column_name])
  data[column_name] = np.where(data[column_name] > upper_bound, upper_bound, data[column_name])

  return data[column_name]


#Define a data wrangle class

class dataWrangler:
 
  def data_handler(self, data):
    self.data = data

    #Treat the marital missing values
    data["marital"] = data["marital"].fillna("unknown")

    marital_status = data["marital"].value_counts()
    status = list(marital_status.index)

    #Treat the age missing values
    median_age ={}
    for i in status:
      mask = data["marital"] == i
      median_age[i] = data[mask]["customer_age"].median()

    for key, value in median_age.items():
      for i in range(len(data)):
        if np.isnan(data["customer_age"][i]):
          if data["marital"][i] == key:
            data["customer_age"][i] = value

    #Treat the missing values in balance column
    data["balance"] = data["balance"].replace(np.nan, data["balance"].median())

    #Treat the missing value in personal loan - replaced it with the column mode
    data["personal_loan"] = data["personal_loan"].replace(np.nan, data["personal_loan"].mode()[0])

    #Replace the missing contact duration with the mean value
    data["last_contact_duration"] = (data["last_contact_duration"]
                                   .replace(np.nan, data["last_contact_duration"].mean()))

    #Replace the missing value with its median
    data["num_contacts_in_campaign"] = (data["num_contacts_in_campaign"]
                                      .replace(np.nan, data["num_contacts_in_campaign"].median()))
  
    #About 85 percent of the data in this column is missing, thus, the column is not statistically significant
    data.drop("days_since_prev_campaign_contact", axis = 1, inplace = True)

    labels = ["18-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61+"]
    bins = [0, 25, 30, 35, 40, 45, 50, 55, 60, np.inf]
    data["customer_age_group"] = pd.cut(data["customer_age"], bins=bins, labels=labels)

    return data

  def preprocess_data(self, data: pd.DataFrame):
    self.data = data
    """The functions preprocesses the partially cleaned data and get it set for clustering.
    Parameters:
    ===========
    Input: DataFrame

    Return
    ======
    Output: DataFrame
    """

    #convert it to object
    data["term_deposit_subscribed"] = data["term_deposit_subscribed"].astype("object")

    #Select all numerical data columns
    numerical_data = data.select_dtypes(include=["float64", "int64"]) 

    #Clip all outliers within the upper and lower bound limit
    cols = numerical_data.columns.to_list()
    for col in cols:
      numerical_data[col] = quantile(numerical_data, col)

    #Standardized the numerical data
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(numerical_data)

    scaled_df = pd.DataFrame(scaled_X, columns=cols)

    return scaled_df
  
  def pca(self, scaled_data):
    """The function transformed the scaled data into 2 dimensional set.
    Parameters:
    ==========
    Input: DataFrame

    Return:
    ======
    Output: DataFrame
    """
    #Instantiate the transformer
    pca = PCA(n_components=2, random_state = 42)

    t_x = pca.fit_transform(scaled_data)

    #convert to dataframe
    x_t_df = pd.DataFrame(t_x, columns=["PC1", "PC2"])

    return x_t_df


#Exploratory Data Analysis
class EDA:

  """Exploratory Data Analysis class. This class calculates of percentage of 
  customers in each feature that bought or would buy bank's product.
  """
  def __init__(self, data: pd.DataFrame):
    self.data = data

  def bal_eda(self, col_name: str, target_col: str):
    bal_eda = self.data.groupby([col_name])[target_col].mean().reset_index(name="Average Balance")
    return bal_eda

  def eda_num(self, col_name: str, target_col: str):
    """Perform EDA on numerical column type
    """
    self.col_name = col_name
    self.target_col = target_col

    grouped = self.data.groupby([col_name, target_col])[target_col].count().reset_index(name="count")

    result = (grouped.groupby(col_name).apply(lambda x: (x.loc[x[target_col] == 1, "count"].iloc[0] 
        / (x.loc[x[target_col] == 0, "count"].iloc[0] + x.loc[x[target_col] == 1, "count"].iloc[0])*100)
        .round(2))).reset_index(name=f"percentage {target_col}")
    return result

  def eda_obj(self, col_name, target_col, k: int):
    """Perform EDA on object column type
    """
    self.col_name = col_name
    self.target_col = target_col
    grouped = self.data.groupby([col_name, target_col])[target_col].count().reset_index(name="count")

    result = (grouped.groupby(col_name).apply(lambda x: (x.loc[x[target_col] == "yes", "count"].iloc[0] 
        / (x.loc[x[target_col] == "no", "count"].iloc[0] + x.loc[x[target_col] == "yes", "count"].iloc[0])*100)
        .round(2))).reset_index(name=f"percentage {target_col} subscription")
    cluster_list = list(range(k))
    res_dict = {}
    for i in cluster_list:
      res_dict[i] = f"Cluster {i}"

    result[col_name] = result[col_name].replace(res_dict)
    
    return result
  
#Graph Builder 

class graphBuilder:

  def bar_chart(self, data, col_x: str, col_y: str):
    colors = ["#3366cc", "#2ca02c", "#d62728", "rgb(217, 95, 2)", "#829dc9", "#9b2cad", "#478b87", "#188f97", "#411897"]
    #Create the plot canvas
    fig = make_subplots( rows = 1, cols = 1,
                    specs = [[{"type": "bar"}]],
                     vertical_spacing = 0.02, horizontal_spacing =0.02,
                    subplot_titles=(f"Cluster {data.columns[1]}", "")
                    )
      
    #Add the bar chart
    fig.add_trace(go.Bar(x= data[col_x], 
                     y= data[col_y],
                     marker = dict(color = colors),
                     orientation = "v"), row = 1, col = 1
              )
    #Style the bar chart
    fig.update_xaxes(title=col_x, visible=True, row=1, col=1)
    fig.update_yaxes(title=col_y, showgrid=True, row=1, col=1)
        

    #Style the layout
    fig.update_layout(width = 550, height=400, bargap=0.2,
                  margin=dict(b=0,r=20,l=20), xaxis=dict(tickmode='linear'),
                  template="plotly_white",
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=False)
    return fig

  def pie_chart(self, data, col_labels: str, col_values: str):

    colors = ["#3366cc", "#2ca02c", "#d62728", "rgb(217, 95, 2)", "#829dc9", "#9b2cad", "#478b87", "#188f97", "#411897"]
    #Create the plot canvas
    fig = make_subplots( rows = 1, cols = 1,
                    specs = [[{"type": "pie"}]],
                     vertical_spacing = 0.02, horizontal_spacing =0.02,
                    subplot_titles=(f"{data.columns[1]}", "")
                    )
      
     #Add the pie chart
    fig.add_trace(go.Pie(labels= data[col_labels].to_list(), 
                     values=data[col_values].to_list(), 
                     textposition="inside", 
                     marker = dict(colors = colors[:len(data[col_labels].to_list())] ), hole = 0.6,
                     hoverinfo = "label+percent", textinfo = "label+percent"), row = 1, col = 1
                    )    

    #Style the layout
    fig.update_layout(width = 400, height=400, bargap=0.2,
                  margin=dict(b=0,r=20,l=20), xaxis=dict(tickmode='linear'),
                  template="plotly_white",
                  title_font=dict(size=18, color='#8a8d93', family="Lato, sans-serif"),
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=False)
    return fig
  
  def cluster_graph(self, labels, x_t_df, k):
    self.labels = labels
    self.x_t_df = x_t_df
    self.k = k

    fig = make_subplots( rows = 1, cols = 1,
                    specs = [[{"type": "scatter"}]],
                     vertical_spacing = 0.02, horizontal_spacing =0.02,
                    subplot_titles=("Customer Segmentation Cluster", "")
                    )
      
    #Add the bar chart

    traces = []
    for cluster in range(k):
      mask = labels == cluster
      trace = go.Scatter(
      x=x_t_df.iloc[mask, 0],
      y=x_t_df.iloc[mask, 1],
      mode='markers',
      name=f'Cluster {cluster}',
      marker=dict(
      size= 12,
      color=cluster)
      )
      traces.append(trace)
    
    fig.add_traces(traces, rows= 1, cols = 1)
    #Style the bar chart
  
    fig.update_yaxes(showgrid=True, row=1, col=1)
    fig.update_xaxes(visible=True, row=1, col=1)     

      #Style the layout
    fig.update_layout(width = 1100, height=450, bargap=0.2,
                  margin=dict(b=0,r=20,l=20), xaxis=dict(tickmode='linear'),
                  template="plotly_white",
                  #title_text = "Customer Segmentation Cluster",
                  xaxis_title=("PC1"),
                  yaxis_title = ("PC2"), 
                  title_font=dict(size=18, color='#8a8d93', family="Lato, sans-serif"),
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=True)
    return fig