#!/usr/bin/env python
# coding: utf-8

# ## GEOG0051 Mining Social and Geographic Datasets
# 
# ## Coursework Part One Mobility Patterns Analysis in Cambridge
# 
# For the first task, you will be analysing the mobility patterns of users from Gowalla, a now-defunct online geo-social networkfrom a decade ago. On Gowalla, users were able to check in at different locations across the course of the day. The dataset that is provided to you (available on Moodle) is a subset of Gowalla users located in Cambridge, UK and, although with some personal identifiers of the users removed, you could trace the movements of particular individuals on certain days, according to their check-ins.
# 
# For further information, the entire dataset is available at https://snap.stanford.edu/data/loc-gowalla.html.

# ## 1 Load data

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
import osmnx as ox
import sklearn
import folium
import nltk
import contextily as ctx


# In[2]:


# read the Gowalla dataset
df = pd.read_csv('Cambridge_gowalla.csv')
len(df)
df.describe()


# ## 2 Visualise individual check-in locations
# 
# Visualise the check-in locations of the GC dataset for users with UserIDs 75027 and 102829 using the Folium library. 
# 
# 1.Comment briefly on your findings of the locations visited by the 2 users, using any library that enables mapping. 
# 
# 2.You should also comment briefly on the privacy implications of this type of analysis.
# 
# Note: This task primarily serves to help familiarise you with the dataset; we advise not to spend too long on it.

# In[3]:


# this filters the dataframe by specific users (UserIDs 75027 and 102829)
# analyze the data attributes of check-in locations
df1 = df[df['User_ID'] == 75027]
df1.describe()
df1.head()


# In[4]:


df2 = df[df['User_ID'] == 102829]
df2.describe()


# In[5]:


# Simple plots of spatial distribution of check-in locations
Lon=df['lon']
Lat=df['lat']

fig,ax = plt.subplots(1,2,figsize=(20,6))

ax[0].scatter(df1.lat,df1.lon,c='g',marker='^',label='UserID 75027')
ax[1].scatter(df2.lat,df2.lon,c='r',marker='*',label='UserID 102829')

ax[0].set_title('UserID 75027')
ax[0].set(xlabel='Longitude(°)',ylabel='Latitude(°)')
ax[0].legend()

ax[1].set_title('UserID 102829')
ax[1].legend()
ax[1].set(xlabel='Longitude(°)',ylabel='Latitude(°)')

fig.suptitle('The spatial distribution of User check-ins for Gowalla',fontsize=20, )
plt.savefig('check-in-simple-plots.png')
plt.show()


# In[6]:


df11 = df1[['lat','lon']]
df11 = df11.values.tolist()

df22=df2[['lat','lon']]
df22 = df22.values.tolist()


# In[7]:


from folium import Map, Marker, Icon, PolyLine

points1 = df11
points2 = df22
# Create a map object with the center as first point in the points list and set the zoom to 17 (you can change it)

my_map = Map(points2[0], zoom_start=17)
# Add markers for each point

for p in points1:
        marker = Marker(p) # Creating a new marker
        icon = Icon(color='green')
        icon.add_to(marker) # Setting the marker's icon color 
        marker.add_to(my_map)

for p in points2:
        marker = Marker(p) # Creating a new marker
        icon = Icon(color='red')
        icon.add_to(marker) # Setting the marker's icon color 
        marker.add_to(my_map)

# Display the map
my_map


# In[8]:


# Save the map
my_map.save('my_map.html')


# # 3 Provide Characterisation of the Gowalla dataset
# 
# Provide a characterisation of the data available for the user 75027 on 30/01/2010 and for user 102829 on 11/05/2010, by visualising the paths for both users using the OSMnx library. Then, summarising your answers in a table in your report andcompute, for each user:
# 
# 1.the maximum displacement (i.e. maximum distance between two consecutive locations they moved between);
# 
# 2.the average displacement (i.e. average distance between two consecutive locations/check-ins);
# 
# 3.the total distance travelled on the day;
# 
# 
# **Note**: All distances should be described in network distance, i.e. the distances of paths along the street networks, rather than geographical distances without consideration of the street paths.
# 

# In[9]:


# read the GC dataset
# user 75027
df3 = df[(df['User_ID']==75027) & (df['date']=='30/01/2010')]
df3.loc[:, 'Time'] = pd.to_datetime(df3.loc[:, 'Time'])
df3.sort_values('Time', inplace=True)
# user 102829
df4 = df[(df['User_ID']==102829) & (df['date']=='11/05/2010')]
df4.loc[:, 'Time'] = pd.to_datetime(df4.loc[:, 'Time'])
df4.sort_values('Time', inplace=True)

print('df3:')
print(df3.head())
print('*' * 20)
print('df4:')
print(df4.head())


# In[112]:


# you can get the graph based on a place and specify the network you would like to get. 
G=ox.graph_from_place('Cambridge,UK', network_type='drive')


# In[38]:


#convert graph to geopandas dataframe
gdf_edges = ox.graph_to_gdfs(G,nodes=False,
                             fill_edge_geometry=True)

# set crs to 3857 (needed for contextily)
gdf_edges = gdf_edges.to_crs(epsg=3857) # setting crs to 3857

# plot edges according to closeness centrality
ax=gdf_edges.plot(figsize=(10,10))

# add a basemap using contextilly
import contextily as ctx
ctx.add_basemap(ax,source=ctx.providers.CartoDB.Positron)
plt.axis('off')
plt.show()


# In[31]:


def get_coordinates_and_get_nearest_node(df, G=G):
    df_nearest_nodes = []
    for lon, lat in zip(df['lon'], df['lat']):
#        nearest_node = ox.get_nearest_node(G, [lat, lon])
        nearest_node = ox.nearest_nodes(G, lon, lat)
        df_nearest_nodes.append(nearest_node)
    return df_nearest_nodes


# get the nearest nodes for users
df3_points = get_coordinates_and_get_nearest_node(df3)
df4_points = get_coordinates_and_get_nearest_node(df4)


# In[32]:


def cal_route(points_list, G=G):
    route_list = [[points_list[0]]]
    for i in range(1, len(points_list)):
        t = nx.shortest_path(G, route_list[-1][-1], points_list[i], weight='length')
        route_list.extend([t])
    return route_list[1:]

# get the shortes path for users

df3_route = cal_route(points_list = df3_points, G = G)
df4_route = cal_route(points_list = df4_points, G = G)


# In[33]:


stats = ox.basic_stats(G)


# In[99]:


gdf1 = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']))


# In[100]:


gdf1 =gdf1.set_crs(epsg=4326)
gdf1 =gdf1.to_crs(epsg=3857)


# In[102]:


gdf1.plot()


# In[ ]:


import geopandas as gpd
route_list = df3_route + df4_route
rc = ["r"] * len(df3_route) + ["b"] * len(df4_route)
fig, ax = ox.plot_graph_routes(G,route_list,route_colors=rc, edge_color='grey', bgcolor='w',
                               edge_alpha=0.8, 
                               node_color='none',edge_linewidth=0.6, 
                               node_size=1, 
                               route_linewidth=2)



ax=gdf1.plot(ax=ax,figsize=(8, 8))
            
#gdf.plot(figsize=(8, 8),ax=ax)
ctx.add_basemap(ax=ax,zoom=12,source=ctx.providers.CartoDB.Positron,crs=gdf1.crs.to_string())


# In[104]:


import geopandas as gpd
route_list = df3_route + df4_route
rc = ["r"] * len(df3_route) + ["b"] * len(df4_route)
fig, ax = ox.plot_graph_routes(G,route_list,route_colors=rc, edge_color='grey', bgcolor='w',
                               edge_alpha=0.8, 
                               node_color='none',edge_linewidth=0.6, 
                               node_size=1, 
                               route_linewidth=2)



ax=gdf1.plot(ax=ax,figsize=(8, 8))
            
#gdf.plot(figsize=(8, 8),ax=ax)
ctx.add_basemap(ax=ax,zoom=12,source=ctx.providers.CartoDB.Positron,crs=gdf1.crs.to_string())


# In[116]:


import geopandas as gpd
route_list = df3_route + df4_route
rc = ["r"] * len(df3_route) + ["b"] * len(df4_route)
graph = ox.plot_graph_routes(G,route_list,route_colors=rc, edge_color='grey', bgcolor='w',
                               edge_alpha=0.8, 
                               node_color='none',edge_linewidth=0.6, 
                               node_size=1, 
                               route_linewidth=2)


# In[125]:


route1=route_list[2]
route1


# In[117]:


#convert graph to geopandas dataframe
nodes,gdf_edges = ox.graph_to_gdfs(graph)

# set crs to 3857 (needed for contextily)
gdf_edges = gdf_edges.to_crs(epsg=3857) # setting crs to 3857

# plot edges according to closeness centrality
ax=gdf_edges.plot(figsize=(10,10))

# add a basemap using contextilly
import contextily as ctx
ctx.add_basemap(ax,source=ctx.providers.CartoDB.Positron)
plt.axis('off')
plt.show()


#ax=gdf1.plot(ax=ax,figsize=(8, 8))
            
#gdf.plot(figsize=(8, 8),ax=ax)
#ctx.add_basemap(ax=ax,zoom=12,source=ctx.providers.CartoDB.Positron,crs=gdf1.crs.to_string())


# In[120]:


graph_map = ox.plot_graph_folium(G, popup_attribute='name', edge_width=2)
route_graph_map = ox.plot_route_folium(G, route_list, route_map=graph_map, popup_attribute='length')
route_graph_map.save('route.html') 


# In[126]:


map_user1 = folium.Map(location=[52.2055314, 0.1186637], zoom_start=15, tiles="cartodbpositron")


fg = folium.FeatureGroup(name='legend name', show=True)
ox.plot_route_folium(G, route1, route_color= 'red ', 
                    route_map=fg)

route_map = map_user1.add_child(fg)


#route_map = ox.plot_route_folium(cambridge_G, route_list, route_map=route_map, route_color='red')


display(route_map)



# In[ ]:


# convert graph to geopandas dataframe
gdf_edges = ox.graph_to_gdfs(G1,nodes=False,
                             fill_edge_geometry=True)

# set crs to 3857 (needed for contextily)
gdf_edges = gdf_edges.to_crs(epsg=3857) # setting crs to 3857

# plot edges according to closeness centrality
ax=gdf_edges.plot('cc',cmap='OrRd',figsize=(10,10))
gdf.plot(ax=ax,markersize=8, color='lightslategrey',aspect=None)
# add a basemap using contextilly
import contextily as ctx
ctx.add_basemap(ax,source=ctx.providers.CartoDB.Positron)
plt.axis('off')
plt.show()


# In[74]:


route_list = df3_route + df4_route
rc = ["r"] * len(df3_route) + ["b"] * len(df4_route)
fig,ax = ox.plot_graph_routes(G,route_list,route_colors=rc, edge_color='grey', bgcolor='w',
                               edge_alpha=0.8, 
                               node_color='none',edge_linewidth=0.6, 
                               node_size=1, 
                               route_linewidth=2)


# add a basemap using contextilly
import contextily as ctx
ctx.add_basemap(ax,zoom=12,source=ctx.providers.CartoDB.Positron)
plt.axis('off')
plt.show()


# In[76]:


fig.savefig("routes.png", bbox_inches='tight', pad_inches=0, dpi=300)


# In[77]:


# calculation of distance 
def cal_distance(points_list, G=G):
    distances = []
    for i in range(0, len(points_list)-1):
        distance = nx.shortest_path_length(G, points_list[i], points_list[i+1],weight='length')
#        print(distance)
        distances.append(distance)
    return distances

df3_shortest_path_distances = cal_distance(df3_points, G)
df4_shortest_path_distances = cal_distance(df4_points, G)


# In[78]:


arr_df3 = np.array(df3_shortest_path_distances)
arr_df4 = np.array(df4_shortest_path_distances)

print(f'the maximum displacement of user [75027]:{np.max(arr_df3)}')
print(f'the minimum displacement of user [75027]:{np.min(arr_df3)}')
print(f'the average displacement of user [75027]:{np.mean(arr_df3)}')
print(f'the total distance of user [75027]:{np.sum(arr_df3)}')
print(f'the route length list of user [75027]:{df3_shortest_path_distances}')
print('*'*30)
print(f'the maximum displacement of user [102829]:{np.max(arr_df4)}')
print(f'the minimum displacement of user [102829]:{np.min(arr_df4)}')
print(f'the average displacement of user [102829]:{np.mean(arr_df4)}')
print(f'the total distance of user [102829]:{np.sum(arr_df4)}')
print(f'the route length list of user [102829]:{df4_shortest_path_distances}')


# ## 4 Comparative analysis of check-in frequencies and network centrality
# Describe the general pattern of user check-ins in the Gowalla dataset in relation to closeness centrality measures for the City of Cambridge, UK, using whatever visual aids you see as fitting to your analysis. 
# 
# Comment on any observable trends which you find most noticeable and/or interesting. 
# 

# **(1) closeness centrality measures for the City of Cambridge, UK**

# In[79]:


# some of the centrality measures are not implemented on multiGraph so first set as diGraph
DG = ox.get_digraph(G)


# In[80]:


# similarly, let's calculate edge closeness centrality: convert graph to a line graph so edges become nodes and vice versa
edge_cc = nx.closeness_centrality(nx.line_graph(DG))


# In[81]:


# set or inscribe the centrality measure of each node as an edge attribute of the graph network object
nx.set_edge_attributes(DG,edge_cc,'cc')
G1 = nx.MultiGraph(DG)


# In[97]:


# add a basemap using contextilly
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
gdf =gdf.set_crs(epsg=4326)
gdf =gdf.to_crs(epsg=3857)


# In[98]:


gdf.crs


# In[105]:


# convert graph to geopandas dataframe
gdf_edges = ox.graph_to_gdfs(G1,nodes=False,
                             fill_edge_geometry=True)

# set crs to 3857 (needed for contextily)
gdf_edges = gdf_edges.to_crs(epsg=3857) # setting crs to 3857

# plot edges according to closeness centrality
ax=gdf_edges.plot('cc',cmap='OrRd',figsize=(10,10))
gdf.plot(ax=ax,markersize=8, color='lightslategrey',aspect=None)
# add a basemap using contextilly
import contextily as ctx
ctx.add_basemap(ax,source=ctx.providers.CartoDB.Positron)
plt.axis('off')
plt.show()


# In[109]:


# convert graph to geopandas dataframe
gdf_edges = ox.graph_to_gdfs(G1,nodes=False,
                             fill_edge_geometry=True)

# set crs to 3857 (needed for contextily)
gdf_edges = gdf_edges.to_crs(epsg=3857) # setting crs to 3857

# plot edges according to closeness centrality
ax=gdf_edges.plot('cc',cmap='OrRd',figsize=(15,15))
gdf.plot(ax=ax,markersize=8, color='lightslategrey',aspect=None)
# add a basemap using contextilly
import contextily as ctx
ctx.add_basemap(ax,zoom=15,source=ctx.providers.CartoDB.Positron)
plt.axis('off')
plt.show()


# In[65]:


# you can then color the edges in the original graph with closeness centrality in the line graph. 
# you can see the area that is the most central here is near Trafalgar square.

nc = ox.plot.get_edge_colors_by_attr(G1, 'cc', cmap='OrRd')
fig, ax = ox.plot_graph(G1, node_size=0, node_color='w', node_edgecolor='gray',bgcolor='w',node_zorder=2,
                        edge_color=nc, edge_linewidth=1.5, edge_alpha=1,show=False,close=False)



#ctx.add_basemap(ax,source=ctx.providers.CartoDB.Positron)
gdf.plot(ax=ax,markersize=8, color='lightslategrey',aspect=None)

#plt.axis('off')
#plt.show()


# **(2) user check-ins frequencies in the Gowalla dataset**

# In[32]:


df_fc = df['User_ID'].value_counts().rename_axis('User_ID').reset_index(name='Total-Check-in')
df_fc.sort_values(by=['User_ID'], inplace=True)
df_fc.head()


# In[33]:


# output the user frequency
df_fc1 = df.User_ID.value_counts() 
df_fc1


# In[34]:


df_fc1.to_csv('df_fc1.csv')


# In[35]:


sns.set_style('darkgrid')
plt.figure(figsize=(25,6))
plt.title('Users check-in frequency', fontsize=17)

sns.lineplot(x=df_fc['User_ID'], y=df_fc['Total-Check-in'])
plt.savefig('Userscheck-infrequency.png')


# In[36]:


df_points = df[['lat','lon']]
df_points = df_points.values.tolist()
df_points


# In[37]:


my_map = Map(df_points[0], zoom_start=13)
# Add markers for each point
for p in df_points:
        marker = Marker(p) # Creating a new marker
        icon = Icon(color='red')
        icon.add_to(marker) # Setting the marker's icon color 
        marker.add_to(my_map)
my_map


# In[38]:


# Save the map
my_map.save('alldata_map.html')


# In[39]:


df_users = get_coordinates_and_get_nearest_node(df)


# In[40]:


location = []
for i in range(len(df)):
    lat = df.iloc[i]['lat']
    lon = df.iloc[i]['lon']
    point = (lat, lon)
    location.append(point)


# In[41]:


counts = {}         
for x in df_users:            
    if x in counts:
        counts[x] += 1
    else:
        counts[x] = 1

for x in list(DG.nodes):
    if x not in df_users:
        counts[x] = 0
    else:
        pass

len(counts)


# In[42]:


import math
for x in counts:
    if counts[x] != 0:
        counts[x] = math.log2(counts[x]+5)
    else:
        pass
for x in counts:
    if counts[x] != 0:
        counts[x] = math.log10(counts[x])
    else:
        pass


# In[43]:


nx.set_node_attributes(DG,name='counts', values=counts)
G_users = nx.MultiGraph(DG)
G_users.nodes(data=True)


# In[69]:


nc = ox.plot.get_node_colors_by_attr(G_users, 'counts',cmap='OrRd')
fig, ax = ox.plot_graph(G_users, node_size=15, 
                        node_color=nc,
                        node_edgecolor='none',
                        edge_color='grey',
                        bgcolor='w',
                        edge_linewidth=0.5, 
                        edge_alpha=1,node_alpha=0.5,node_zorder=0)

#ctx.add_basemap(ax,zoom=12,source=ctx.providers.CartoDB.Positron)
#plt.axis('off')
#plt.show()
# add the basemap
ctx.add_basemap(ax=ax, zoom=20, source=ctx.providers.Stamen.TonerLite,alpha=0.8)

ax.set_title("Prediction via MultiNB for Venue Stars Level in the City of Calgary, Canada",fontsize= 20)

# this removes the axis
ax.set_axis_off()

# this tightens the layout
fig.tight_layout()


# In[45]:


node_cc = nx.closeness_centrality(DG)
# set the attributes back to its edge
nx.set_node_attributes(DG, node_cc,'cc')
# and turn back to multiGraph for plotting
Gn = nx.MultiGraph(DG)
nc = ox.plot.get_node_colors_by_attr(Gn, 'cc', cmap='OrRd')
fig, ax = ox.plot_graph(Gn, node_size=10, node_color=nc, 
                        node_edgecolor='none', bgcolor='w',
                        edge_color='grey', edge_linewidth=0.5, 
                        edge_alpha=1,node_alpha=0.6,node_zorder=0)
fig.savefig("users_closeness.png", bbox_inches='tight', pad_inches=0, dpi=300)


# In[ ]:


plt.savefig("CC.png", bbox_inches='tight', pad_inches=0, dpi=300)


# In[ ]:


ofile = f'C:/Users/zexir/Desktop/4.25-mining/1Mobility Patterns Analysis in Cambridge/check-ins.png'
fig.savefig(ofile)


# In[85]:


get_ipython().system('pip install folium')
import folium
from folium import plugins
from folium.plugins import HeatMap


# In[93]:


#1 star of entire dataset inc predictions

middle_lat = df["lat"].median()
middle_lon = df["lon"].median()


map_1star = folium.Map([middle_lat, middle_lon], zoom_start=10, tiles="cartodbpositron")
map_1star_pred = folium.Map([middle_lat, middle_lon], zoom_start=10, tiles="cartodbpositron")


# In[94]:



heat = df[['lat', 'lon']]
heat.describe()


heat_data = [[row['lat'],row['lon']] for index, row in heat.iterrows()]

HeatMap(heat_data).add_to(map_1star)


map_1star


# In[ ]:




