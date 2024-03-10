from dash import Dash,dcc,  html,Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
# Importing required libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import nltk
import warnings
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import reuters
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# from sklearn.preprocessing.label import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_json(r'D:\Prx_Project\ss_clsteranalysis\V10_All_From_MongoDb_v1.json',lines=True)
df1=df

# ----------collecting authors data
aut_data = []
for author in df1.values:
    #     print(author[1])
    for aut in author[2]:
        #         print(aut)
        aut_data.append([aut, author[1]])

# ----------collecting book title data

title_data=[]
for author in df1.values:
    title_data.append([author[4],author[1]])

# -----creation authors dataframe & title dataframe
df_clu=pd.DataFrame(aut_data,columns=['author','abstract'])
df_title=pd.DataFrame(title_data,columns=['title','abstract'])

# ---removing the duplicate authors details
df_clu.drop_duplicates(inplace=True)
# df_title.drop_duplicates(inplace=True)

df_clu.sort_values("author",ascending=True)
df_title.sort_values("title",ascending=True)

df_clu.author.value_counts()
df_title.title.value_counts()
# df_clu.author=='Wei Wang']

def simple_tokenizer(text):
    nltk.download('stopwords')
    stopWords = stopwords.words('english')
    charfilter = re.compile('[a-zA-Z]+')
    nltk.download('punkt')
    # tokenizing the words:
    words = word_tokenize(text)
    # converting all the tokens to lower case:
    words = map(lambda word: word.lower(), words)
    # let's remove every stopwords
    words = [word for word in words if word not in stopWords]
    # stemming all the tokens
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    ntokens = list(filter(lambda token: charfilter.match(token), tokens))
    return ntokens


def author_cluster(author_name):
    df_aut_one=df_clu[df_clu.author==author_name]
    df_aut_one.dropna(inplace=True)
    abstr=df_aut_one.abstract.tolist()
    tf_idf_vector = TfidfVectorizer(tokenizer = simple_tokenizer, max_features = 1000, norm = 'l2')
    from sklearn.model_selection import train_test_split

    transformed_vector = tf_idf_vector.fit_transform(abstr)
    clustering = SpectralClustering(n_clusters=5, assign_labels="discretize", random_state=0)
    prediction = clustering.fit_predict(transformed_vector)
    labels = clustering.labels_
    pred_table=pd.DataFrame(abstr)
    pred_table['labels']=labels
    num=np.arange(1,len(labels)+1)

    s = pd.Series(num)
    pred_table['index']=s.astype('int')
    serieslst=pred_table.values.tolist()
    newseries = []
    for i in serieslst:
        newseries.append([i[1], i[2]])
    lables = pred_table.labels.tolist()
    labels_new = list(set(lables))
    G=nx.Graph()
    G.add_nodes_from(labels_new,width=6)
    G.add_edges_from(newseries,width=1)
    nx.draw(G,with_labels=True,node_color='g')
    plt.show()


def title_cluster(title_name):
    # ---collecting given title aritical
    df_title_one=df_title[df_title.title==title_name]
    df_title_one.dropna(inplace=True)

    # ---------------collecting abstract details for analysis
    abstr=df_title_one.abstract.tolist()
    print("-------------abstr",abstr)
    if len(abstr)<=1:
        abstr=abstr+abstr

    # -----converting abstract text to numberic for semantic analysis
    tf_idf_vector = TfidfVectorizer(tokenizer = simple_tokenizer, max_features = 1000, norm = 'l2')

    # -----collecting stop word and removing from the data text
    nltk.download('stopwords')
    stopWords = stopwords.words('english')
    charfilter = re.compile('[a-zA-Z]+')

    nltk.download('punkt')
    transformed_vector = tf_idf_vector.fit_transform(abstr)


    # -------applying the spectralcluster analysis, default cluster as 5
    clustering = SpectralClustering(n_clusters=5, assign_labels="discretize", random_state=0)

    # generating predictions for the clusters
    prediction = clustering.fit_predict(transformed_vector)

    # creating a data frame with cluster labels and predictions
    labels = clustering.labels_
    pred_table=pd.DataFrame(abstr)
    pred_table['labels']=labels
    num=np.arange(1,len(labels)+1)

    s = pd.Series(num)
    pred_table['index']=s.astype('int')
    serieslst=pred_table.values.tolist()
    newseries = []
    for i in serieslst:
        newseries.append([i[1], i[2]])

    # generating the network graph for title

    G=nx.Graph()
    G.add_edges_from(newseries)
    nx.draw(G)
    plt.show()



def author_title_cluster(author_name,title_name):
    # ---collecting given title aritical
    df_title_one=df_title[df_title.title==title_name]
    df_title_one.dropna(inplace=True)
    df_clu_one = df_clu[df_clu.author == author_name]
    df_clu_one.dropna(inplace=True)
    df_aut_title=pd.concat([df_title_one,df_clu_one],axis=0)
    print(df_aut_title.head())


    print("-----df_title_one", df_title_one.shape)
    print("-----df_clu_one", df_clu_one.shape)
    print("-----df_aut_title", df_aut_title.shape)

    # ---------------collecting abstract details for analysis
    abstr=df_aut_title.abstract.tolist()
    print("-------------abstr",abstr)
    if len(abstr)<=1:
        abstr=abstr+abstr

    # -----converting abstract text to numberic for semantic analysis
    tf_idf_vector = TfidfVectorizer(tokenizer = simple_tokenizer, max_features = 1000, norm = 'l2')

    # -----collecting stop word and removing from the data text
    nltk.download('stopwords')
    stopWords = stopwords.words('english')
    charfilter = re.compile('[a-zA-Z]+')

    nltk.download('punkt')
    transformed_vector = tf_idf_vector.fit_transform(abstr)


    # -------applying the spectralcluster analysis, default cluster as 5
    clustering = SpectralClustering(n_clusters=5, assign_labels="discretize", random_state=0)

    # generating predictions for the clusters
    prediction = clustering.fit_predict(transformed_vector)

    # creating a data frame with cluster labels and predictions
    labels = clustering.labels_
    pred_table=pd.DataFrame(abstr)
    pred_table['labels']=labels
    num=np.arange(1,len(labels)+1)

    s = pd.Series(num)
    pred_table['index']=s.astype('int')
    serieslst=pred_table.values.tolist()
    newseries = []
    for i in serieslst:
        newseries.append([i[1], i[2]])

    # generating the network graph for title

    G=nx.Graph()
    G.add_edges_from(newseries)
    nx.draw(G)
    plt.show()


# author_cluster('Edwin R. Hancock')

# title_cluster('String searching algorithms')
# #
# author_title_cluster('Edwin R. Hancock','String searching algorithms')


app = Dash (__name__)
app.layout=html.Div([
    html.Header("My Cluter Analysis"),
    dcc.Dropdown(id = "Author",
                 options=df_clu.author.unique(),
                 value='Edwin R. Hancock'),
    dcc.Graph(id ="my_networkx")
])

@app.callback(Output('my_networkx',"figure"),
              Input('Author','value')
              
              )
def sync_input(author):
    author_cluster(author)
    



if __name__== "__main__":
    app.run_server(debug=False)