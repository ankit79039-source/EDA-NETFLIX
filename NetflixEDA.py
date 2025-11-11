
import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('netflix.csv')

df.head(3)
df.tail(4)

df.columns

df.shape

df.info()


df.describe()

df.describe(include='object')

df.describe(include='all')


# In[21]:


df.sample()


# In[22]:


df.dtypes


# In[23]:


df[df.duplicated()]


# ## 🖌️Data Wrangling

# ### 📚Unnesting the columns (directors , casts , countrys , listed_in)

# In[24]:


unnesting = ['director', 'cast', 'listed_in','country']
for column in unnesting:
    df[column] = df[column].str.split(', ')
    df = df.explode(column)


# In[25]:


df.shape


# In[26]:


df.dtypes


# In[27]:


df.reset_index(drop=True,inplace=True)


# In[28]:


df


# #### 👉Insights : 
# Note after unnesting the no.of movies and TvShows rows are increased but still we use nunique() instead of counts to get the accurate data...
# - Movie      - 6131
# - TV Show    - 2676
# - Total      - 8807

# #### Treating Nulls

# In[29]:


plt.figure(figsize=(14,8))
sns.heatmap(df.isnull())
plt.title('Visual Check of Nulls',fontsize=20)
plt.show()


# In[30]:


df.isna().sum().sort_values(ascending=False)


# In[31]:


for i in df.columns:
    null_pct = (df[i].isna().sum() / df.shape[0]) *100
    if null_pct > 0 :
        print(f'Null_pct of {i} is {round(null_pct,3)} %')


# In[32]:


df[df.date_added.isna()]


# In[33]:


df['date_added'] = pd.to_datetime(df['date_added'] ,format="%B %d, %Y" , errors='coerce')


# In[34]:


df['date_added'].fillna(df['date_added'].mode()[0], inplace=True)


# In[35]:


df.isna().sum().sort_values(ascending=False)


# In[36]:


df.dtypes


# In[37]:


df['year_added'] = df['date_added'].dt.year


# In[38]:


df.sample()


# In[39]:


df.dtypes


# In[40]:


df.shape


# In[41]:


df.info()


# In[42]:


df.isna().sum().sort_values(ascending=False)


# In[43]:


df[df.rating.isna() | df.duration.isna()]


# In[44]:


df["country"].fillna("Unknown",inplace=True)
df["cast"].fillna("Unknown actors",inplace=True)
df["director"].fillna("Unknown director",inplace=True)
df["rating"].fillna("Unknown",inplace=True)


# In[45]:


df.isna().sum()


# In[46]:


df[df.duration.isna()]


# In[47]:


df.rating.value_counts()


# In[48]:


rvc = df.rating.value_counts(dropna=False).reset_index()


# In[49]:


plt.figure(figsize=(14,5))
a = sns.barplot(rvc , x='rating' , y='count' , color='red' , width=0.3)
plt.title('Raw analysis of Ratings',fontsize=20,fontweight='bold')
a.bar_label(a.containers[0], label_type='edge')
plt.show()


# In[50]:


df[df.director=='Louis C.K.'] 


# In[51]:


# df['duration'] = df.loc[df.director=='Louis C.K.'].replace(np.nan , df.rating) didn't work
# df['duration'].fillna(df['rating'], inplace=True) # this works .. easy method .. but make sure the required rows alone change
df.loc[df['director']=='Louis C.K.', 'duration']=df.loc[df['director']=='Louis C.K.','duration'].fillna(df.loc[df['director'] == 'Louis C.K.', 'rating'])
# b4 line end => .fillna(df.loc[df['director'] == 'Louis C.K.', 'rating'])


# In[52]:


df[df.director=='Louis C.K.'] 


# In[53]:


df.loc[df['director'] == 'Louis C.K.', 'rating'] = 'Unknown'


# In[54]:


df[df.director=='Louis C.K.'] 


# In[55]:


df.shape


# In[56]:


df.dtypes


# In[57]:


df.isna().sum()


# ##### 📂we will segregate the data into movie data and tv-shows data and fill the duration appropriately...

# In[58]:


df.type.value_counts()


# In[59]:


movies_data = df[df.type=='Movie']


# In[60]:


movies_data.shape


# In[61]:


tvshows_data = df[df.type=='TV Show']


# In[62]:


tvshows_data.shape


# In[63]:


movies_data.sample()


# In[64]:


movies_data.isna().sum()


# In[65]:


tvshows_data.sample()


# In[66]:


tvshows_data.isna().sum()


# In[67]:


movies_data['runtime_in_mins'] = movies_data['duration'].str.split(' ').str[0]
tvshows_data['no_of_seasons'] = tvshows_data['duration'].str.split(' ').str[0]


# In[68]:


movies_data.sample()


# In[69]:


movies_data.dtypes


# In[70]:


movies_data.runtime_in_mins = movies_data.runtime_in_mins.astype(int)


# In[71]:


movies_data.dtypes


# In[72]:


movies_data = movies_data.drop(columns=['description','duration']).reset_index(drop=True)


# In[73]:


movies_data.shape


# In[ ]:


# movies data is done


# In[ ]:


# tvshows.... 


# In[74]:


tvshows_data.tail()


# In[75]:


tvshows_data.no_of_seasons.value_counts()


# In[76]:


tvshows_data.dtypes


# In[77]:


tvshows_data.no_of_seasons = tvshows_data.no_of_seasons.astype(int)


# In[78]:


tvshows_data.no_of_seasons.dtypes


# In[79]:


tvshows_data = tvshows_data.drop(columns=['description','duration']).reset_index(drop=True)


# In[80]:


tvshows_data.sample(3)


# In[81]:


df = df.drop(columns=['description']).reset_index(drop=True)


# In[82]:


print(f'Cleaned Netflix data has {df.shape[0]} Rows and {df.shape[1]} Columns')
print(f'Netflix Movies data has {movies_data.shape[0]} Rows and {movies_data.shape[1]} Columns')
print(f'Netflix TV shows data has {tvshows_data.shape[0]} Rows and {tvshows_data.shape[1]} Columns')


# In[83]:


df.shape


# In[84]:


df.type.value_counts()


# In[85]:


plt.figure(figsize=(25,8), layout='tight').suptitle('Visual checks of Nulls',fontsize=20,fontweight="bold",fontfamily='serif')


plt.subplot(1,3,1)
sns.heatmap(df.isnull())
plt.title('cleaned Netflix data Nulls',fontsize=12)
plt.xlabel('metrics',fontsize=12)
plt.ylabel('row_numbers',fontsize=12)

plt.subplot(1,3,2)
sns.heatmap(movies_data.isnull())
plt.title('Movies data Nulls',fontsize=12)
plt.xlabel('metrics',fontsize=12)
plt.ylabel('row_numbers',fontsize=12)


plt.subplot(1,3,3)
sns.heatmap(tvshows_data.isnull())
plt.title('Tv Shows data Nulls',fontsize=12)
plt.xlabel('metrics',fontsize=12)
plt.ylabel('row_numbers',fontsize=12)

plt.show()


# #### 👉Insights:
# - 👆🏽 Red color indicates data has 0 % nulls in all columns

# In[86]:


# saving the files for further analysis:

df.to_csv('netflix_cleaned_data.csv',sep=',',index=False)
movies_data.to_csv('cleaned_movies_data.csv',sep=',',index=False)
tvshows_data.to_csv('cleaned_tvshows_data.csv',sep=',',index=False)


# # Exploratory Data Analysis (EDA):

# In[87]:


nx = pd.read_csv('netflix_cleaned_data.csv')
md = pd.read_csv('cleaned_movies_data.csv')
tvd = pd.read_csv('cleaned_tvshows_data.csv')


# #### 📌Q. How are contents distributed in Netflix Platform ?

# In[88]:


pg = nx.groupby('type')['show_id'].nunique()
pg


# In[89]:


pgdf = pg.reset_index()
pgdf


# In[90]:


plt.figure(figsize=(13.5, 4))
font = {'fontweight': 'bold', 'family': 'serif'}
plt.suptitle("Netflix Contents Distribution", fontdict=font, fontsize=20)
explode = tuple(0.08 if i == 0 else 0 for i in range(len(pg)))
plt.style.use('seaborn-v0_8-bright')

plt.subplot(1, 2, 1)
plt.pie(pg, labels=pg.index, startangle=80, explode=explode,
        colors=['red', '#dedede'][:len(pg)], shadow=True,
        autopct='%1.1f%%', textprops={'color': "k"})

plt.subplot(1, 2, 2)
ax = sns.countplot(x='type', data=pgdf, palette=['red', '#dedede'][:len(pgdf['type'].unique())])
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10)
sns.despine(left=True, bottom=True)
ax.set_ylabel('')
ax.set_yticks([])
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()


# #### 👉Insights : 
# 
# - we can clearly interpret that nearly *70%* contents are `Movies` whereas *30%* are `Tvshows` contents in **Netflix** content library.
# 

# #### 📌Q. Outliers check: 

# In[91]:


plt.figure(figsize=(16,6))
font = {'weight':'bold',
        'family':'serif'}
plt.suptitle("Netflix Contents Distribution",fontweight='bold',fontsize=20)

plt.style.use('seaborn-v0_8-bright')

plt.subplot(1,2,1)
sns.violinplot(nx,x='type',y='release_year',palette=['red','#dedede'])
sns.despine()
plt.xlabel('')
plt.title(" Violin Distribution",fontdict=font,fontsize=14)

plt.subplot(1,2,2)
sns.boxplot(nx,x='type',y='release_year',palette=['red','#dedede'])
sns.despine()
plt.xlabel('')
plt.title("Box & Whisker Distribution",fontdict=font,fontsize=14)

plt.show()


# #### 👉Insights : 
# 
# - Here we can easily identify that the contents released before 1960 are the outliers.
# --------------------------------

# #### 📌Q. In which year maximum contents got released ?

# In[92]:


ryvc = nx.release_year.value_counts()[:20]


# In[93]:


plt.figure(figsize=(13,5))
plt.style.use('seaborn-v0_8-bright')
sns.countplot(nx , y='release_year' , order = ryvc.index , palette=['red','dimgrey'] , width=0.2)
sns.despine(bottom=True)
plt.xticks([])
plt.xlabel('')
plt.title('Years With Maximum contents Released',fontsize=16,fontweight='bold',fontfamily='serif')
plt.show()


# #### 👉Insights :
# 
# - Netflix began enlarging their contents library from 2000 and acquired maximum contents so far in `2018` followed by `2019`&`2017`. 
# --------------------------------

# #### 📌Q. What are the top 15 countries consumption of movies and tvshows ? 

# In[94]:


# countrywise content count with movies_data
cm = md.groupby('country')[['show_id']].nunique().sort_values(by='show_id',ascending=False)
cm = cm[:15]
cwm = cm[cm.index!=('Unknown')]
cwm


# In[95]:


# countrywise content count with tvshows_data
ctv = tvd.groupby('country')[['show_id']].nunique().sort_values(by='show_id',ascending=False)
cwtv = ctv[:15]
cwtv = cwtv[cwtv.index!=('Unknown')]
cwtv


# In[96]:


# Graphical Analysis
plt.figure(figsize=(16,6))
plt.suptitle('Countries consuming Movies & TV Shows',
             fontsize=20,fontweight="bold",fontfamily='serif')
plt.style.use('seaborn-v0_8-bright')

c1 = sns.barplot(cwm, x=cwm.index , y=cwm.show_id,
                 color='red' , width=0.4 , label='Movies_count')
#c1.bar_label(c1.containers[0], label_type='edge',color='r')
plt.xlabel('Country',fontsize=12)
plt.ylabel('Content count',fontsize=12)
plt.legend(loc='upper right')
plt.xticks(rotation=30)

c2 = sns.barplot(cwtv, x=cwtv.index , y=cwtv.show_id,
                 color='dimgray' , width=0.2 , label='Tvshows_count')
plt.xlabel('Country',fontsize=12)
plt.ylabel('Content count',fontsize=12)
plt.legend(loc='upper right')
plt.xticks(rotation=30)

top_n = 14
for i in range(top_n):
    c1.annotate(cwm.show_id[i], (i+0.12, cwm.show_id[i]+50),
                ha='left', va='baseline',color='red')
    
for i in range(top_n):
    c2.annotate(cwtv.show_id[i], (i+0.22, cwtv.show_id[i]),
                ha='left', va='baseline', color='dimgrey')
    
plt.show()


# #### 👉Insights :
# 
# - The top 5 countries with the highest count of Movies and TV Shows are aiding in recognizing the key players in TV show production and Movies production.
# - We can infer that `US` , `India` , `UK` , `France` , `Canada` , `Japan` are the top entertainment consumers while other countries significantly contribute to the OTT content library.
# 
# ----------------------------------
# 

# #### 📌Q. How much contents are added every year in netflix ? 

# In[97]:


yc = nx.groupby(['year_added','type'])[['show_id']].nunique().reset_index()
yc.sort_values(by='show_id',ascending=False)


# In[98]:


yc['show_id'].sum()


# In[99]:


ycm = md.groupby(['year_added','type'])[['show_id']].nunique().reset_index()
ycm


# In[107]:


ycm['show_id'].sum()


# In[100]:


yctv = tvd.groupby(['year_added','type'])[['show_id']].nunique().reset_index()
yctv


# In[101]:


yctv['show_id'].sum()


# **MULTIVARIATE ANALISYS: Type by year_added** 

# In[102]:


plt.figure(figsize=(16,6))
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')
c = sns.barplot(data = yc, x = 'year_added' , y = 'show_id' ,
                hue = 'type', palette=['red','dimgrey'] , width=0.35)
plt.title('Contents added to Netflix Yearwise',
          fontsize=16,fontweight="bold",fontfamily='serif')
c.bar_label(c.containers[0], label_type='edge',color='red')
c.bar_label(c.containers[1], label_type='edge',color='dimgray')
plt.legend(loc='upper left')
plt.show()


# #### 👉Insights :
# **Netflix**'s content acquisition strategy over time.
# - The bar plot shows the distribution of content added to Netflix across different years.
# - There appears to be an increasing trend in the total number of content items added to Netflix over the years.
# - The bars tend to get taller as you move from left to right, suggesting that Netflix has been continuously expanding its content library.
# - The variation in bar heights from year to year highlights how Netflix's content strategy has evolved. 
# - Some years show significant spikes, while others have lower counts, indicating variations in content acquisition.
# --------------------------------------

# #### 📌Q. How much contents gets released every year ?

# In[103]:


mr = md.groupby('release_year')[['title']].nunique()
mr = mr.reset_index()
mr


# In[104]:


mr.title.sum()


# In[105]:


tvr = tvd.groupby('release_year')[['title']].nunique()
tvr = tvr.reset_index()
tvr


# In[106]:


tvr.title.sum()


# **UNIVARIATE ANALISYS**

# In[107]:


plt.figure(figsize=(16,6))
plt.style.use('seaborn-v0_8-darkgrid')
sns.lineplot(data=mr , x='release_year' , y='title' , color='r' ,
             label = 'Movies', marker='d')
sns.lineplot(data=tvr , x='release_year' , y='title' , color='dimgrey',
             label='Tv Shows' , marker='d')
plt.title('Contents Released count Yearwise',fontsize=16,
                      fontweight="bold",fontfamily='serif')
plt.ylabel('contents uploaded count')
plt.legend(loc='upper left')
plt.show()


# In[108]:


plt.figure(figsize=(30,10) , dpi=250)
plt.suptitle('Yearly Release of Contents',
             fontsize=20,fontweight="bold",fontfamily='serif')
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')

mr= mr[mr.release_year>2000]
plt.subplot(1,2,1)
c = sns.barplot(mr , x = 'release_year' , y='title', color='tomato',width=0.98)
c.bar_label(c.containers[0], label_type='edge',color='r')
sns.pointplot(mr , x='release_year' , y='title' , color='r')
plt.xlabel("Year",fontsize=12)
plt.ylabel("Movies Counts", fontsize=12)
plt.title("Year of Movies Release", fontsize=16,fontweight="bold",fontfamily='serif')

plt.subplot(1,2,2)
d = sns.histplot(x = tvr.release_year, bins = 10, kde = True, 
             color='dimgrey' , edgecolor ='dimgrey')
d.bar_label(d.containers[0], label_type='edge',color='dimgrey')
plt.xlabel('Year',fontsize=12)
plt.ylabel("TV Show Counts", fontsize=12)
plt.title("Year of TV show Release", fontsize=16,fontweight="bold",fontfamily='serif')

plt.show()


# In[109]:


mr.dtypes


# In[110]:


plt.figure(figsize=(30,10) , dpi=250)
plt.suptitle('Yearly Release of Contents',
             fontsize=20,fontweight="bold",fontfamily='serif')
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')

plt.subplot(1,2,1) 
sns.boxplot(md , x= 'release_year', color='red')
sns.despine()
plt.title('Movie Releases',fontsize=16,fontfamily='serif')

plt.subplot(1,2,2)
sns.boxplot(tvr , x= 'release_year', color='dimgrey')
sns.despine(left=True)
plt.title('Tvshow Releases',fontsize=16,fontfamily='serif')

plt.show()


# In[111]:


sns.jointplot(nx , x='year_added' , y='release_year' , hue='type' , 
                      palette=['red','dimgrey'])
plt.show()


# #### 👉Insights :
# 
# - The plot uses two distinct lines in different colors (red for movies and dimgrey for TV shows) to enable a side-by-side comparison.
# - The plot reveals that Movies has been more dominant in terms of release counts in any given year and in the recent past audience focus shifts on watching web series. 
# - The dominance of movies and TV shows over the years can indicate changing audience preferences, industry trends and Netflix's strategic decisions.
# - Netflix has been continuously expanding its content library, offering more choices to its subscribers.
# ---------------

# #### 📌Q. How the contents genre segregated ? 

# In[112]:


mg = md.groupby(['listed_in'])[['title']].nunique().sort_values(by='title',ascending=False)
mg = mg.reset_index()
mg


# In[113]:


tvg = tvd.groupby(['listed_in'])[['title']].nunique().sort_values(by='title',ascending=False)
tvg = tvg.reset_index()
tvg


# In[114]:


plt.figure(figsize=(25,10))
plt.suptitle('Popular Genre Contents count',fontsize=20,
             fontweight="bold",fontfamily='cursive')
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')

plt.subplot(1,2,2)
sns.barplot(mg , x='title' , y='listed_in' , color='red' , width=0.3)
sns.despine(left=True,bottom=True,trim=True)
plt.title('Movie Genre',fontsize=16)
plt.xlabel('Movies count')
plt.xticks([])
n=20
for i in range(n):
    plt.annotate(mg.title[i], (mg.title[i]+75,i+0.2),
                 ha='center' , va='bottom' , color='r')

plt.subplot(1,2,1)
sns.barplot(tvg , x='title' , y='listed_in' , color='dimgrey' , width=0.3)
sns.despine(left=True,bottom=True,trim=True)
plt.title('Tv Show Genre',fontsize=16)
plt.xlabel('TvShows count')
plt.xticks([])
nn=22
for i in range(nn):
    plt.annotate(tvg.title[i], (tvg.title[i]+45,i+0.2),
                 ha='center' , va='bottom' , color='dimgrey')

plt.show()


# **Genre-WordCloud**

# In[115]:


from wordcloud import WordCloud


# In[116]:


plt.figure(figsize=(16,4))
plt.suptitle('Popular Genre Contents in Word Cloud',
             fontsize=16,fontweight="bold",fontfamily='fantasy')
plt.style.use('default')
plt.style.use('dark_background')

plt.subplot(1,2,1)
mgwc = WordCloud(width=1600, height=800, background_color='black',
                 colormap='Reds').generate(md.listed_in.to_string())
plt.imshow(mgwc)
plt.axis('off')
plt.title("Movie Genre",fontsize=14,fontweight='bold',fontfamily='serif')

plt.subplot(1,2,2)
tvgwc = WordCloud(width=1600, height=800, background_color='black',
                  colormap='Greys').generate(tvd.listed_in.to_string())
plt.imshow(tvgwc)
plt.axis('off')
plt.title("Tv Shows Genre",fontsize=14,fontweight='bold',fontfamily='serif')

plt.show()


# #### 👉Insights :
# 
# - The plot illustrates the popularity of various genres in Movies and TV Shows on the platform.
# - Here, we can see that `Hollywood contents` , `Dramas` , `Comedies` are the Top and Evergreen genres
# - The plot provides insights into audience preferences, indicating which genres are more prevalent in Movies and TV Shows.
# -----------

# #### 📌Q. what genre's are more preferred by directors ?

# In[117]:


mdgc = md.groupby('listed_in')['director'].nunique().sort_values(ascending=False)
mdgc


# In[118]:


tvdgc = tvd.groupby('listed_in')['director'].nunique().sort_values(ascending=False)
tvdgc


# In[119]:


plt.figure(figsize=(25, 12))
plt.suptitle('Directors popular Genre Contents',
                fontsize=20,fontweight="bold",fontfamily='cursive')
plt.style.use('seaborn-v0_8-bright')

plt.subplot(1,2,1)
a = sns.barplot(y=mdgc.index, x=mdgc.values, color='r',width=0.3)
a.bar_label(a.containers[0], label_type='edge',color='r')
plt.title('Movie Directors comfy Genre\'s',fontsize=20,
                  fontweight="bold",fontfamily='serif')
sns.despine(left=True,bottom=True,trim=True)
plt.ylabel('Genre')
plt.xticks([])

plt.subplot(1, 2, 2)

a = sns.barplot(
    y=tvdgc.index,
    x=tvdgc.values,
    color='dimgrey',
    width=0.3
)

a.bar_label(
    a.containers[0],
    label_type='edge',
    color='dimgrey'
)

plt.title(
    "TvShow Directors Comfy Genre's",
    fontsize=20,
    fontweight="bold",
    fontfamily='serif'
)

sns.despine(left=True, bottom=True, trim=True)

plt.ylabel('')
plt.xticks([])

plt.show()


# #### 👉Insights:
# 
# - The diversity of genre that the directors are more comfortable indicates that Netflix has content on all genres in its library.
# - The top genres with the most directors are International Movies, Dramas, Comedies, Documentaries, Independent Movies, and Action & Adventure.
# ----------

# #### 📌Q. What are genres more preferred in each country ?  

# In[121]:


plt.figure(figsize=(20, 8))
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')


given_country = input("Enter your preferred Choice of Country : ")

mcountry = md[md["country"] == given_country]
tvcountry = tvd[tvd["country"] == given_country]

mc_data = mcountry.groupby(['listed_in','type'])[['show_id']].nunique()
mc_data = mc_data.sort_values(by=["show_id"],ascending = False).reset_index()

mtv_data = tvcountry.groupby(['listed_in','type'])[['show_id']].nunique()
mtv_data = mtv_data .sort_values(by=["show_id"],ascending = False).reset_index()


plt.suptitle('Genre Distribution across the selected Country'
             ,fontsize=20,fontweight="bold",fontfamily='serif')

plt.subplot(1,2,1)
a = sns.barplot(mc_data , y='listed_in', x='show_id',color='red',width = 0.2)
a.bar_label(a.containers[0], label_type='edge',color='r')
plt.title('Movie Genre Distribution')
plt.xlabel('')
plt.xticks([])
plt.ylabel('Count of Contents')
plt.xticks(rotation=90, ha = 'center',fontsize = 8)
plt.yticks(fontsize =8)

plt.subplot(1,2,2)
b = sns.barplot(mtv_data , y='listed_in', x='show_id',color='dimgrey',width = 0.2)
b.bar_label(b.containers[0], label_type='edge',color='dimgrey')
sns.despine(bottom=True,left=True)
plt.title('TvShows Genre Distribution')
plt.xlabel('')
plt.xticks([])
plt.ylabel('')
plt.xticks(rotation=90, ha = 'center',fontsize = 8)
plt.yticks(fontsize =8)

plt.show()


# #### 📌Q. How are contents distributed based on Runtime & Seasons ?

# In[122]:


tvd.groupby(['no_of_seasons'])[['title']].nunique().sum()


# In[123]:


md.groupby(['runtime_in_mins'])[['title']].nunique().sum()


# In[124]:


mrt = md.groupby(['runtime_in_mins'])[['title']].nunique().sort_values(by='title',ascending=False)
mrt = mrt.reset_index()
mrt


# In[125]:


tvs = tvd.groupby(['no_of_seasons'])[['title']].nunique().sort_values(by='title',ascending=False)
tvs = tvs.reset_index()
tvs


# In[126]:


plt.figure(figsize=(25,13))
plt.suptitle('Length of Contents',
             fontsize=20,fontweight="bold",fontfamily='serif')
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')

plt.subplot(2,1,1)
sns.lineplot(mrt , y='title' , x='runtime_in_mins' , color='red' , marker='d')
sns.despine()
plt.grid(True, linestyle='--', alpha=0.6)
plt.title('Movie runtime\n' ,fontsize=12,fontweight="bold")
plt.ylabel('Movies count')
plt.text(205,123,'It is seen that the most optimum\nduration for a content is',
         fontsize=14,fontfamily='sans-serif')
plt.text(248,122,'90-120 Minutes',color='r',
         fontsize=14,fontfamily='fantasy',fontweight='bold')
max_value = mrt.title.max()
max_x = mrt[mrt.title == max_value]['runtime_in_mins'] 
sns.scatterplot(x=max_x, y=max_value, color='#dedede', marker='s', s=10000)

plt.subplot(2,1,2)
sns.barplot(tvs , y='title' , x='no_of_seasons' , color='dimgrey' , width=0.3)
sns.despine()
plt.title('------------------------------\n Tv Shows Seasons',
          fontsize=12,fontweight="bold")
plt.ylabel('TvShows count')
n=15
for i in range(n):
     plt.annotate(tvs.title[i], (i+0.2,tvs.title[i]+45),
                  ha='center' , va='bottom' , color='dimgrey')

plt.show()


# In[127]:


plt.figure(figsize=(15,8))
plt.suptitle('Length of Contents',
             fontsize=20,fontweight="bold",fontfamily='serif')
plt.style.use('default')
plt.style.use('seaborn-v0_8-darkgrid')

plt.subplot(1,2,1)
sns.histplot(x = md.runtime_in_mins, bins = 200, color='red',
            kde = True, edgecolor = 'salmon')
plt.xlabel("Movie Duration in mins",fontsize=12)
plt.ylabel("Movies Counts", fontsize=12)
plt.title("Duration of Movies", fontsize=14)

plt.subplot(1,2,2)
b = sns.histplot(x = tvd.no_of_seasons, bins = 10, kde = True, 
             color='dimgrey' , edgecolor ='k')
b.bar_label(b.containers[0], label_type='edge',color='dimgrey')
plt.xlabel('No.of Seasons',fontsize=12)
plt.ylabel("TV Show Counts", fontsize=12)
plt.title("Duration of TV shows", fontsize=14)

plt.show()


# In[128]:


plt.figure(figsize=(18,8) , dpi=250)
plt.suptitle('Contents Duration',
             fontsize=20,fontweight="bold",fontfamily='serif')
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')

plt.subplot(2,2,1)
sns.boxplot(md , x ='runtime_in_mins', color='red',showfliers=True)
plt.title('Movies',fontsize=16,fontfamily='serif')

plt.subplot(2,2,3)
sns.boxplot(mrt , x='title', color='red')

plt.subplot(2,2,2)
sns.boxplot(tvd , x= 'no_of_seasons', color='dimgrey')
plt.title('TvShows',fontsize=16,fontfamily='serif')

plt.subplot(2,2,4)
sns.boxplot(tvs , x='title', color='dimgrey')
sns.despine(left=True,trim=True)

plt.show()


# In[129]:


# #### 👉Insights :
# 
# - The majority of movies appear to have a runtime around **90-120** `minutes` . This is evident from the peak in the red line plot having highlighted the maximum value (maximum movie count) using a large silver square marker
# - In the TV shows, there are a higher number of TV shows with a smaller number of seasons (e.g., **1-3** `seasons`), and the counts gradually decrease as the number of seasons increases.
# ------------

# #### 📌Q. What are the ratings given for the contents uploaded on netflix ?

# In[140]:


md.groupby(['rating'])[['title']].nunique().sum()


# In[141]:


tvd.groupby(['rating'])[['title']].nunique().sum()


# In[142]:


movie_rating = md.groupby(['rating'])[['title']].nunique().reset_index()
movie_rating = movie_rating.sort_values(by='title',ascending=False)
movie_rating


# In[143]:


tv_rating = tvd.groupby(['rating'])[['title']].nunique().reset_index()
tv_rating = tv_rating.sort_values(by='title',ascending=False)
tv_rating


# In[144]:


plt.figure(figsize=(16,6))
plt.style.use('ggplot')
sns.lineplot(data=movie_rating , x='rating' , y='title' , color='r' , label = 'Movies', marker='s')
sns.lineplot(data=tv_rating , x='rating' , y='title' , color='dimgrey', label='Tv Shows' , marker='o')
plt.title('Ratings of the Contents Released',fontsize=16,fontweight="bold",fontfamily='serif')
plt.ylabel('contents uploaded count')
plt.legend(loc='upper right')
plt.show()


# #### 👉Insights :
# *MOVIES*
# - The most common content rating is "TV-MA," with a total of 2,062 contents , typically associated with content intended for mature audiences.
# - "TV-14" is the second most common rating, with 1,427 content count indicating content suitable for viewers aged 14 and older.
# - "Restricted: R - Under 17 requires accompanying parent or adult guardian" is the third most common rating, with 797 titles.
# 
# *TV SHOWS*
# - The "TV-MA" rating with 1,145 titles suggests that a significant portion of the content is intended for mature audiences.
# - "TV-14" is the second most common rating, with 733 titles indicates that contents are for viewers aged 14 and older.
# - "TV-PG" - parental guidance is recommended stands third with 323 contents in Tv programs.
# ------------

# #### 📌Q. Diversify the actors with more contents ?

# In[145]:


movies_cast = md.groupby('cast')[['title']].nunique().sort_values(by='title',ascending=False)[1:20]
movies_cast


# In[146]:


tv_cast = tvd.groupby('cast')[['title']].nunique().sort_values(by='title',ascending=False)[1:20]
tv_cast


# In[147]:


plt.figure(figsize=(20,12))
plt.suptitle('Actors with more Contents',
             fontsize=20,fontweight="bold",fontfamily='serif',color='k')
plt.style.use('Solarize_Light2')

plt.subplot(2,1,1)
c1 = sns.barplot(movies_cast, y=movies_cast.index , x='title',color='red',width=0.3)
sns.despine(left=True,bottom=True,trim=True)
plt.title('Actors with more Movie Contents',
          fontsize=16,fontweight="bold",fontfamily='serif',color='r')
plt.xticks([])
plt.yticks(fontweight='bold')
plt.xlabel('')
plt.ylabel('Actors',fontsize=12)
for i in range(19):
    c1.annotate((str(movies_cast.title[i])+' movies'), (movies_cast.title[i]+1,i+0.3),
                ha='center' , va='bottom' , color='red')

plt.subplot(2,1,2)
c2 = sns.barplot(tv_cast, y=tv_cast.index , x='title',color='dimgrey',width=0.3)
sns.despine(left=True,bottom=True,trim=True)
plt.title('\n Actors with more TvShows Contents',
          fontsize=16,fontweight="bold",fontfamily='serif',color='dimgray')
plt.xticks([])
plt.xlabel('')
plt.yticks(fontweight='bold')
plt.ylabel('Actors',fontsize=12)
for i in range(19):
    c2.annotate((str(tv_cast.title[i])+' shows'), (tv_cast.title[i]+0.63,i+0.3),
                ha='center' , va='bottom' , color='dimgrey')

plt.show()


fmd = md.groupby('director')[['show_id']].nunique()
fmd = fmd.sort_values(by='show_id',ascending=False)[1:21]
fmd


# In[149]:


ftvd = tvd.groupby(['director'])[['show_id']].nunique()
ftvd= ftvd.sort_values(by='show_id',ascending=False)[1:21]
ftvd


# In[150]:


plt.figure(figsize=(20,12))
plt.suptitle('Directors with more Contents',
             fontsize=20,fontweight="bold",fontfamily='serif',color='k')
plt.style.use('Solarize_Light2')

plt.subplot(2,1,1)
c1 = sns.barplot(fmd, y=fmd.index , x='show_id',color='red',width=0.3)
sns.despine(left=True,bottom=True,trim=True)
plt.title('Directors with more Movie Contents',
          fontsize=16,fontweight="bold",fontfamily='serif',color='r')
plt.xticks([])
plt.yticks(fontweight='bold')
plt.xlabel('')
plt.ylabel('Directors',fontsize=12)
for i in range(20):
    c1.annotate((str(fmd.show_id[i])+' movies'), (fmd.show_id[i]+0.53,i+0.3),
                ha='center' , va='bottom' , color='red')

plt.subplot(2,1,2)
c2 = sns.barplot(ftvd, y=ftvd.index , x='show_id',color='dimgrey',width=0.3)
sns.despine(left=True,bottom=True,trim=True)
plt.title('\n Directors with more TvShows Contents',
          fontsize=16,fontweight="bold",fontfamily='serif',color='dimgray')
plt.xticks([])
plt.xlabel('')
plt.yticks(fontweight='bold')
plt.ylabel('Directors',fontsize=12)
for i in range(20):
    if ftvd.show_id[i]>1:
        c2.annotate((str(ftvd.show_id[i])+' shows'),(ftvd.show_id[i]+0.07,i+0.3),
                    ha='center' , va='bottom' , color='dimgrey')
    else:
        c2.annotate((str(ftvd.show_id[i])+' show'),(ftvd.show_id[i]+0.07,i+0.3),
                    ha='center' , va='bottom' , color='dimgrey')
plt.show()



ad = nx[['cast','show_id','director','type']]
ad = ad[ad.cast!='Unknown actors']
ad = ad[ad.director!='Unknown director']
ad = ad.drop_duplicates().reset_index(drop=True)
ad


# In[152]:


nad = ad.groupby(['cast','director','type'])[['show_id']].nunique()
new_ad = nad.reset_index().sort_values(by='show_id',ascending=False)
new_ad['ad_pair'] = new_ad['cast']+'-'+new_ad['director']
new_ad


# In[153]:


mad = new_ad[new_ad.type=='Movie']
tvad = new_ad[new_ad.type=='TV Show']


# In[154]:


mad.info()


# In[155]:


mad = mad[['ad_pair','show_id']]
mad


# In[156]:


tvad = tvad[['ad_pair','show_id']]
tvad


# In[157]:


mad.dtypes


# In[158]:


fmad = mad[:25].set_index('ad_pair')
ftvad = tvad[:25].set_index('ad_pair')


# In[159]:


fmad


# In[178]:


ftvad


# In[160]:


plt.figure(figsize=(19, 15))
plt.suptitle('Actor - Director pairs',fontsize=20,
                 fontweight="bold",fontfamily='serif')
plt.style.use('default')
plt.style.use('Solarize_Light2')

plt.subplot(2,1,1)
a1 = sns.barplot(y=fmad.index, x=fmad.show_id, color='red',width=0.3)
plt.title('Movie Directors-Actors Combo',fontsize=12,fontweight="bold")
sns.despine(left=True,bottom=True,trim=True)
plt.yticks(fontweight='bold')
plt.xticks([])
plt.xlabel('No.of times worked together')
for i in range(25):
    a1.annotate((str(fmad.show_id[i])+' times'), (fmad.show_id[i]+0.47,i+0.5),
                ha='center' , va='bottom' , color='red')

plt.subplot(2,1,2)
a2 = sns.barplot(ftvad , y=ftvad.index, x=ftvad.show_id, color='dimgrey',width=0.3)
plt.title('TvShow Directors-Actors Combo',fontsize=12,fontweight="bold")
sns.despine(left=True,bottom=True,trim=True)
plt.yticks(fontweight='bold')
plt.xticks([])
plt.xlabel('No.of times worked together')
for i in range(25):
    if ftvad.show_id[i]>1:
        a2.annotate((str(ftvad.show_id[i])+' times'), (ftvad.show_id[i]+0.07,i+0.2),
                ha='center' , va='bottom' , color='dimgrey')
    else:
        a2.annotate((str(ftvad.show_id[i])+' time'), (ftvad.show_id[i]+0.07,i+0.3),
                ha='center' , va='bottom' , color='dimgrey')
    
plt.show()



md.columns


# In[162]:


movie_release = md[['show_id','title','date_added']]
movie_release = movie_release.reset_index(drop=True)
movie_release


# In[163]:


movie_release.dtypes


# In[164]:


movie_release['date_added'] = pd.to_datetime(movie_release['date_added'])


# In[165]:


movie_release.dtypes


# In[166]:


movie_release.isna().sum()


# In[167]:


movie_release['week_uploaded'] = movie_release['date_added'].dt.isocalendar().week
movie_release['uploaded_weekday'] = movie_release['date_added'].dt.strftime('%A')
movie_release['uploaded_month'] = movie_release['date_added'].dt.strftime('%B')


# In[168]:


month_order = ['January', 'February', 'March', 'April', 'May',
               'June', 'July', 'August', 'September', 
               'October', 'November', 'December']
movie_release['uploaded_month']= pd.Categorical(movie_release['uploaded_month'],
                                            categories=month_order, ordered=True)


# In[169]:


movie_release


# In[170]:


week_movie_release=movie_release.groupby('week_uploaded')['show_id'].nunique()
week_movie_release=week_movie_release.reset_index()
week_movie_release


# In[171]:


week_movie_release.sum()


# In[172]:


monthly_movie_release=movie_release.groupby('uploaded_month')['show_id'].nunique()
monthly_movie_release = monthly_movie_release.reset_index()
monthly_movie_release = monthly_movie_release.sort_values(by='uploaded_month')
monthly_movie_release.reset_index(drop=True)
monthly_movie_release


# In[173]:


monthly_movie_release = movie_release.groupby('uploaded_month')['title'].nunique().reset_index()


# In[174]:


movies_release_pivot = movie_release.pivot_table(index='uploaded_month', 
                                                 columns='uploaded_weekday', 
                                                 values='show_id', 
                                                 aggfunc=pd.Series.nunique)

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
             'Friday', 'Saturday','Sunday']
movies_release_pivot = movies_release_pivot[day_order]
movies_release_pivot


# In[175]:


plt.figure(figsize=(15, 10))
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')
sns.heatmap(movies_release_pivot, cmap='Reds',
                annot=True, fmt='d' , linewidth=0.1)
plt.title("Movie Releases by Weekday and Month",
              fontfamily='serif',fontsize=16,fontweight='bold')
plt.tick_params(axis='both', which='both', left=False, bottom=False)
plt.show()


# In[176]:


movies_release_pivot.sum().sort_values(ascending=False)


# In[177]:


movies_release_pivot.sum().sum()


# #### 📌Q. What is the best time to launch a Tvshow ?

# In[178]:


tvs_release = tvd[['show_id','title','date_added']]
tvs_release = tvs_release.reset_index(drop=True)
tvs_release


# In[179]:


tvs_release.dtypes


# In[180]:


tvs_release.isna().sum()


# In[181]:


tvs_release['date_added'].fillna(tvs_release['date_added'].mode()[0],inplace=True)


# In[182]:


tvs_release['date_added'] = pd.to_datetime(tvs_release['date_added'])


# In[183]:


tvs_release['date_added'].dtypes


# In[184]:


tvs_release['week_uploaded'] = tvs_release['date_added'].dt.isocalendar().week
tvs_release['uploaded_weekday'] = tvs_release['date_added'].dt.strftime('%A')
tvs_release['uploaded_month'] = tvs_release['date_added'].dt.strftime('%B')


# In[185]:


month_order = ['January', 'February', 'March', 'April', 'May',
               'June', 'July', 'August', 'September', 
               'October', 'November', 'December']
tvs_release['uploaded_month']= pd.Categorical(tvs_release['uploaded_month'],
                                    categories=month_order, ordered=True)


# In[186]:


tvs_release


# In[187]:


tvs_release.groupby('week_uploaded')['show_id'].nunique().sum()


# In[188]:


week_release = tvs_release.groupby('week_uploaded')['show_id'].nunique()
week_release = week_release.reset_index()
week_release


# In[189]:


# Ensure uploaded_month exists
tvs_release['uploaded_month'] = tvs_release['uploaded_month'].astype(str)

# Ensure show_id is string
tvs_release['show_id'] = tvs_release['show_id'].astype(str)

# Groupby
month_release = (
    tvs_release.groupby('uploaded_month')['show_id']
    .nunique()
    .reset_index()
)

month_release


# In[190]:


month_release.show_id.sum()


# In[191]:


tvs_release_pivot = tvs_release.pivot_table(
            index='uploaded_month' , 
            columns='uploaded_weekday' , 
            values='show_id' , 
            aggfunc=pd.Series.nunique
    ) 

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                 'Friday', 'Saturday','Sunday']
tvs_release_pivot = tvs_release_pivot[day_order]
tvs_release_pivot


# In[192]:


tvs_release_pivot.sum(axis=1)


# In[193]:


tvs_release_pivot.sum().sort_values(ascending=False)


# In[194]:


tvs_release_pivot.sum().sum()


# In[195]:


plt.figure(figsize=(15, 10))
plt.style.use('seaborn-v0_8-bright')
sns.heatmap(tvs_release_pivot, cmap='Greys', annot=True, 
                fmt='d' , linewidth=0.1)
plt.title("TvShows Releases by Weekday and Month",
                  fontfamily='serif',fontsize=16,fontweight='bold')
plt.tick_params(axis='both', which='both', left=False, bottom=False)
plt.show()


# In[201]:


for df in [week_movie_release, monthly_movie_release, week_release, month_release]:
    if 'count' not in df.columns:
        cols = df.columns.tolist()
        if len(cols) >= 2:
            df.rename(columns={cols[1]: 'count'}, inplace=True)

plt.figure(figsize=(30,15))
plt.suptitle('Releases Broader View', fontfamily='serif', fontsize=20, fontweight='bold')

plt.subplot(2,2,1)
sns.pointplot(data=week_movie_release, x='week_uploaded', y='count', color='r')
plt.title('Movie Weekly Releases Count', fontfamily='serif', fontsize=16, fontweight='bold')
plt.xlabel('Week Number')
plt.ylabel('No. of Movies Released')

plt.subplot(2,2,2)
sns.pointplot(data=week_release, x='week_uploaded', y='count', color='dimgrey')
plt.title('TV Shows Weekly Releases Count', fontfamily='serif', fontsize=16, fontweight='bold')
plt.xlabel('Week Number')
plt.ylabel('No. of TV Shows Released')

plt.subplot(2,2,3)
sns.pointplot(data=monthly_movie_release, x='uploaded_month', y='count', color='r')
plt.title('Movie Monthly Releases Count', fontfamily='serif', fontsize=16, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('No. of Movies Released')

plt.subplot(2,2,4)
sns.pointplot(data=month_release, x='uploaded_month', y='count', color='dimgrey')
plt.title('TV Shows Monthly Releases Count', fontfamily='serif', fontsize=16, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('No. of TV Shows Released')

plt.show()



plt.figure(figsize=(20,11) , dpi=400)
plt.suptitle('Releases by Weekday and Month',fontfamily='serif',
                     fontsize=20,fontweight='bold')
plt.style.use('seaborn-v0_8-bright')

plt.subplot(1,2,1)
sns.heatmap(movies_release_pivot, cmap='Reds', annot=True, 
                    fmt='d' , linewidth=0.1)
plt.title("Movie Releases by Weekday and Month",fontfamily='serif',
                  fontsize=16,fontweight='bold')
plt.yticks(rotation=0)
plt.tick_params(axis='both', which='both', left=False , bottom=False)

plt.subplot(1,2,2)
sns.heatmap(tvs_release_pivot, cmap='Greys', annot=True, fmt='d' ,
                    linewidth=0.1)
plt.title("TvShows Releases by Weekday and Month",fontfamily='serif',
                  fontsize=16,fontweight='bold')
plt.yticks(rotation=0)
plt.tick_params(axis='both', which='both', left=False, bottom=False)

plt.show()


# In[203]:


sns.jointplot(md , x='release_year' , y='year_added' , color='red')
plt.text(1940,2021.6,'Movies Comparison' , color='dimgrey' , fontsize=12 , fontweight='bold')
plt.show()


# In[204]:


sns.jointplot(tvd , x='release_year' , y='year_added' , color='dimgrey')
plt.text(1930,2021.4,'TvShows Comparison',color='red',fontsize=12,fontweight='bold')
plt.show()




filtered_md = md[['show_id','title','release_year','year_added']].drop_duplicates()


# In[206]:


filtered_md.shape


# In[207]:


filtered_md.sample()


filtered_md['time_diff_in_yrs']=filtered_md['year_added']-filtered_md['release_year']


# In[209]:


filtered_md.head()


# In[210]:


filtered_md.time_diff_in_yrs.mode()[0]



filtered_md['time_diff_in_yrs'].value_counts()


# In[212]:


fmdg=filtered_md.groupby(['time_diff_in_yrs'])[['title']].agg(numbers_released = ('title','count'))
# .agg(numbers_released = ('title','count'))


# In[213]:


rtf = fmdg.sort_values(by='numbers_released',ascending=False)


# In[214]:


rtf = rtf.reset_index()


# In[215]:


rtf


# In[216]:


rtf[rtf.time_diff_in_yrs==-1]


# In[217]:


filtered_md[filtered_md.time_diff_in_yrs==-1]


filtered_tvd = tvd[['show_id','title','release_year','year_added']].drop_duplicates()


# In[219]:


filtered_tvd


# In[220]:


filtered_tvd.shape


# In[221]:


filtered_tvd['time_diff_in_yrs']=filtered_tvd['year_added']-filtered_tvd['release_year']


# In[222]:


filtered_tvd.tail()


# In[223]:


filtered_tvd['time_diff_in_yrs'].mode()


# In[224]:


filtered_tvd['time_diff_in_yrs'].mode()[0]


# #### 👉Insights:
# 
# - Time difference is `ZERO` indicating that the contents are added to the netflix library within the same year.
# - The contents are added to the Netflix OTT platform within some months or days of release.
# - Now a days , as per aggrements the new content will be uploaded in OTT platforms within `24 hours` after aired on Television.

# In[225]:


ftv = filtered_tvd.groupby(['time_diff_in_yrs'])[['title']].agg(numbers_released = ('title','count'))
# .agg(numbers_released = ('title','count'))


# In[226]:


rtv = ftv.sort_values(by='numbers_released',ascending=False)


# In[227]:


rtv


# In[228]:


rtv = rtv.reset_index()


# In[229]:


rtv[(rtv.time_diff_in_yrs==-1)|(rtv.time_diff_in_yrs==-2)|(rtv.time_diff_in_yrs==-3)]


# In[230]:


filtered_tvd[filtered_tvd.time_diff_in_yrs<0]


# #### 👉Insights:
# 
# - Contents like TV show episodes and web series Seasons are mostly released directly in OTT platform.
# - From the day and time of Airing any contents, those will be uploaded on Netflix library within `24 Hours`.

# In[233]:


plt.figure(figsize=(25,12))
plt.suptitle('Comparision of TimeFrame Gap for contents getting uploaded on Netflix',
             fontfamily='serif', fontweight='bold', fontsize=20)
plt.style.use('default')
plt.style.use('ggplot')

plt.subplot(2,1,1)
sns.pointplot(data=rtf, x='time_diff_in_yrs', y='numbers_released', color='r')
plt.title('Movies uploaded in Netflix TimeFrame (Years)',
          fontfamily='serif', fontweight='bold', fontsize=16)
plt.text(43,1450,
         "Mode of time_diff is the uploading factor\n"
         "and it is evident that the contents\n"
         "are uploaded at the earliest possible",
         fontsize=12, fontfamily='cursive')
plt.text(43,1150,
         "It is seen that movies now a days are being released\n"
         "in OTT platforms with the time gap of",
         fontsize=12, fontfamily='cursive')
plt.text(54.2,1030,
         "50-60 days (4 weeks).",
         fontsize=16, fontfamily='fantasy', fontweight='bold', color='red')

plt.subplot(2,1,2)
sns.pointplot(data=rtv, x='time_diff_in_yrs', y='numbers_released', color='dimgrey')
plt.title('TvShows uploaded in Netflix TimeFrame (Years)',
          fontfamily='serif', fontweight='bold', fontsize=16)
plt.text(27,1100,
         "Mode of time_diff is the uploading factor\n"
         "and it is evident that the contents\n"
         "are uploaded at the earliest possible",
         fontsize=12, fontfamily='cursive')
plt.text(27,950,
         "It is seen that TvShows now a days are being released\n"
         "in OTT platforms with the time gap of",
         fontsize=12, fontfamily='cursive')
plt.text(34.5,840,
         "24 Hrs from airing on Television",
         fontsize=16, fontfamily='fantasy', fontweight='bold', color='dimgrey')

plt.show()



nx.release_year.dtypes


# In[235]:


cu = nx.copy
cu = nx.drop_duplicates(subset='show_id')
cu['date_added'] = pd.to_datetime(cu['date_added'])
cu['release_date'] = pd.to_datetime(cu['release_year'].astype(str))
cu


# In[236]:


cu['days_to_add'] = (cu['date_added'] - cu['release_date']).dt.days
cu


# In[237]:


# considering entire data (both tvshows and movies)
day_mode = cu['days_to_add'].mode()[0]
day_mode 


# In[238]:


cu.dtypes


# In[239]:


# filtering the recent past data (after 2018) for movies
fcum = cu[(cu.release_year>2018) & (cu.type=='Movie')]
fcum


# In[240]:


upload_date_interval_movie = fcum['days_to_add'].mode()[0]
upload_date_interval_movie


# In[241]:


# filtering the recent past data (after 2018) for tvshows
fcutv = cu[(cu.release_year>2018) & (cu.type=='TV Show')]
fcutv


# In[242]:


fcutv['days_to_add'].value_counts()


# In[243]:


upload_date_interval_tvs = fcutv['days_to_add'].mode()
upload_date_interval_tvs


# In[244]:


upload_date_interval_tvs.mean()


# In[245]:


upload_date_interval_tvs.median()


# In[246]:


upload_date_interval_tvs[0]


# In[247]:


plt.figure(figsize=(15,8))
plt.suptitle('TimeFrame Gap for contents getting uploaded on Netflix',
             fontfamily='serif',fontweight='bold',fontsize=20)
plt.style.use('default')
plt.style.use('seaborn-v0_8-whitegrid')

plt.subplot(1,2,1)
plt.hist(fcum['days_to_add'], bins=30, color='red', edgecolor='white')
plt.title('Days to Add to Netflix After Theatrical Movie Release',
         fontfamily='serif',fontweight='bold',fontsize=12)
plt.xlabel('Days to Add')
plt.ylabel('Frequency')
plt.axvline(upload_date_interval_movie, color='k', linestyle='-.',
            linewidth=3, label=f'Mode: {upload_date_interval_movie} days')
plt.legend(fontsize=16)

plt.subplot(1,2,2)
plt.hist(fcutv['days_to_add'], bins=30, color='silver', edgecolor='white')
plt.title('Days to Add to Netflix After Tvshow Air',
         fontfamily='serif',fontweight='bold',fontsize=12)
plt.xlabel('Days to Add')
plt.ylabel('Frequency')
plt.axvline(upload_date_interval_tvs[0], color='red', linestyle='-.',
            linewidth=3, label=f'Mode: {upload_date_interval_tvs[0]} days')
plt.legend(fontsize=16)

sns.despine()
plt.show()


md.columns


# In[249]:


#Shortest Movie
shortest_movie = md.loc[(md['runtime_in_mins']==np.min(md.runtime_in_mins))] [['title','runtime_in_mins']].drop_duplicates()
shortest_movie


# In[250]:


#Longest Movie
longest_movie = md.loc[(md['runtime_in_mins']==np.max(md.runtime_in_mins))] [['title','runtime_in_mins']].drop_duplicates()
longest_movie


# #### 📌Q. Find how are the contents added to Netflix library (uploading rate)?

# In[251]:


md.sample()


# In[252]:


df = md.drop_duplicates(subset='title')


# In[253]:


df.shape


# In[254]:


df = df[['show_id','title','date_added']]


# In[255]:


df.shape


# In[256]:


df.sample()


# In[257]:


df.dtypes


# In[258]:


df['date_added'] = pd.to_datetime(df['date_added'])


# In[259]:


df.dtypes


# In[260]:


df['year_added'] = df['date_added'].dt.year


# In[261]:


df['month_added'] = df['date_added'].dt.month_name()


# In[262]:


df.sample()


# In[263]:


upload_rate = df.groupby('year_added')['month_added'].value_counts()


# In[264]:


upload_rate


# In[265]:


month_order = ['January', 'February', 'March', 'April', 'May',
               'June', 'July','August', 'September', 
               'October', 'November', 'December']
upload_rate = upload_rate.unstack()[month_order]
upload_rate


# In[266]:


upload_rate = upload_rate.fillna(0)
upload_rate


# In[267]:


plt.figure(figsize=(16,8) , dpi=500)
plt.style.use('default')
plt.style.use('seaborn-v0_8-bright')
sns.heatmap(upload_rate, cmap='Reds', edgecolors='beige', linewidths=2)
plt.title('Monthly Netflix Contents Update Rate',
          fontsize=16, fontfamily='calibri', fontweight='bold')
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.xlabel('Uploaded Month',fontsize=12)
plt.ylabel('Uploaded Year',fontsize=12)
plt.show()


# In[268]:


sns.jointplot(upload_rate)
plt.show()



sns.pairplot(nx)
plt.show()


# 🏷️**PAIRPLOT OF MOVIES Data**

# In[276]:


sns.pairplot(md)
plt.show()


# 🏷️**PAIRPLOT OF TvSHOWS Data**

# In[277]:


sns.pairplot(tvd)
plt.show()


# 🏷️**Pair plotting of Upload Rate**

# In[278]:


sns.pairplot(upload_rate, kind='scatter')
plt.show()


# In[475]:


sns.pairplot(movies_release_pivot)
plt.show()


# In[476]:


sns.pairplot(tvs_release_pivot)
plt.show()


# In[477]:


movies_release_pivot.corr()


# In[484]:


tvs_release_pivot.corr()


# In[512]:


plt.figure(figsize=(25,10))
plt.suptitle('Correlation',fontsize=30,fontfamily='serif',fontweight='bold')

plt.subplot(1,2,1)
sns.heatmap(movies_release_pivot.corr() ,cmap='Reds',annot=True)
plt.title('Movies corr',fontsize=16,fontweight='bold')
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.subplot(1,2,2)
sns.heatmap(movies_release_pivot.corr() ,cmap='Greys',annot=True)
plt.title('Tvshows corr',fontsize=16,fontweight='bold')
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.show()


# #### 👉Insights:
# 
# - The correlation are found to nominal and on an average it is found that data has a `Positive Correlation` on weekly uploading rate

# In[507]:


mdc = md[['release_year','year_added','runtime_in_mins']].corr()
mdc


# In[508]:


tvdc = tvd[['release_year','year_added','no_of_seasons']].corr()
tvdc


# In[511]:


plt.figure(figsize=(15,5))
plt.suptitle('Correlation',fontsize=20,fontfamily='serif',fontweight='bold')

plt.subplot(1,2,1)
sns.heatmap(mdc,cmap='Reds',annot=True)
plt.title('Movies corr',fontsize=16,fontweight='bold')
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.subplot(1,2,2)
sns.heatmap(tvdc,cmap='Greys',annot=True)
plt.title('Tvshows corr',fontsize=16,fontweight='bold')
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.show()

