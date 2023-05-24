# EDA and Customer Segmentation Dashboard Design

Description
===========
This is a domain specific prototype design work in fulfilment of an assignment in CETM 46 Course module. I designed an EDA and Customer Segmentation Dashboard that is capable of clustering customers with similar behaviour together, perform exploratory analysis on each cluster and make recommendations. The target domain is banking sector. The tailored recommendations for each cluster made by the app, will help banks to boost their revenue by closing more sales successfully.


Installation
=============
To install the application, kindly follow the steps below:
1. Extract the zip files into a root folder 
2. Change directory and move into the root folder
3. Run the python file app.py
4. Copy the localhost url into a browser and press enter, eg: copy 127.0.0.1:8050.

Python Libraries Required
=========================
* Python 3.7.3
* Numpy - 1.21.5
* Pandas - 1.3.5
* Scikit-learn - 0.21.3
* Dash - 2.0.0
* Dash-bootstrap-components - 1.0.2
* Dash-core-components - 2.0.0
* Dash-html-components - 2.0.0
* Dash-table - 5.0.0
* Pydantic - 1.10.4
* Matplotlib - 3.3.3

Alternatively, you can run the requirements.txt in a terminal within the root folder using this command **_pip install -r requirements.txt_** to install all the required libraries.

Use Case
========
By default, the app selects 6 - the optimal value of cluster's number, and segment customers with similiar behavior in each of these groups. It also performs exploratory analysis and visualizes the clusters term deposit subscription and personal loan subscription using a bar-chart and a pie-chart respectively. And finally makes a recommendation on the likely product to offer the top three clusters.

You can select the cluster number from the slider, and the app will interactively recluster the groups based on the selected number, perform EDA, visualize and make recommendations. 