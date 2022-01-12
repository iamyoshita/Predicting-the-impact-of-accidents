import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import seaborn
import re
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('US_Accidents_Dec20.csv')
print("This data has the following number of rows and cols:\n",(df.shape))
print("count of rows based on severity",df["Severity"].value_counts())

df = df.drop(['ID','Source','TMC', 'End_Time', 'End_Lat', 'End_Lng', 'Distance(mi)','Description','Number','County','Wind_Chill(F)'], axis=1)


# used df["col name"].unique().size to check if any col has only one value and dropped it
df = df.drop(['Country','Turning_Loop'], axis=1)

#drop all duplicate data
df.drop_duplicates(inplace=True)

#drop rows with NaN, we have more than enough data even after deleting these rows
df = df.dropna(subset=['City','Zipcode','Timezone','Airport_Code','Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)',
                       'Sunrise_Sunset'])


#aggregation, store mean for missing values in precipitation
df['Precipitation_NA'] = 0
df.loc[df['Precipitation(in)'].isnull(),'Precipitation_NA'] = 1
df['Precipitation(in)'] = df['Precipitation(in)'].fillna(df['Precipitation(in)'].mean())
df.loc[:5,['Precipitation(in)','Precipitation_NA']]

#new features from start time
df["Start_Time"] = pd.to_datetime(df["Start_Time"])

# get day,month,year,weekday,hour and minute 
df["Day"] = df["Start_Time"].dt.day
df["Month"] = df["Start_Time"].dt.month
df["Year"] = df["Start_Time"].dt.year
df["Weekday"] = df["Start_Time"].dt.weekday
df["Hour"] = df["Start_Time"].dt.hour
df["Minute"] = df["Start_Time"].dt.minute
df = df.drop(['Start_Time'], axis=1)

#wind direction
df.loc[df['Wind_Direction']=='Calm','Wind_Direction'] = 'CALM'
df.loc[(df['Wind_Direction']=='West')|(df['Wind_Direction']=='WSW')|(df['Wind_Direction']=='WNW'),'Wind_Direction'] = 'W'
df.loc[(df['Wind_Direction']=='South')|(df['Wind_Direction']=='SSW')|(df['Wind_Direction']=='SSE'),'Wind_Direction'] = 'S'
df.loc[(df['Wind_Direction']=='North')|(df['Wind_Direction']=='NNW')|(df['Wind_Direction']=='NNE'),'Wind_Direction'] = 'N'
df.loc[(df['Wind_Direction']=='East')|(df['Wind_Direction']=='ESE')|(df['Wind_Direction']=='ENE'),'Wind_Direction'] = 'E'
df.loc[df['Wind_Direction']=='Variable','Wind_Direction'] = 'VAR'

#weather condition abstraction 
weather ='!'.join(df['Weather_Condition'].dropna().unique().tolist())
weather = np.unique(np.array(re.split(
    "!|\s/\s|\sand\s|\swith\s|Partly\s|Mostly\s|Blowing\s|Freezing\s", weather))).tolist()
df['Clear'] = np.where(df['Weather_Condition'].str.contains('Clear', case=False, na = False), 1, 0)
df['Cloud'] = np.where(df['Weather_Condition'].str.contains('Cloud|Overcast', case=False, na = False), 1, 0)
df['Rain'] = np.where(df['Weather_Condition'].str.contains('Rain|storm', case=False, na = False), 1, 0)
df['Heavy_Rain'] = np.where(df['Weather_Condition'].str.contains('Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms', case=False, na = False), 1, 0)
df['Snow'] = np.where(df['Weather_Condition'].str.contains('Snow|Sleet|Ice', case=False, na = False), 1, 0)
df['Heavy_Snow'] = np.where(df['Weather_Condition'].str.contains('Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls', case=False, na = False), 1, 0)
df['Fog'] = np.where(df['Weather_Condition'].str.contains('Fog', case=False, na = False), 1, 0)
df = df.drop(['Weather_Timestamp','Weather_Condition','Precipitation(in)'], axis=1)

#POI features
##POI_features = ['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal']
##
##fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(15, 10))
##
##plt.subplots_adjust(hspace=0.5,wspace = 0.5)
##for i, feature in enumerate(POI_features, 1):    
##    plt.subplot(3, 4, i)
##    seaborn.countplot(x=feature, hue='Severity', data=df, palette="Set2")
##    
##    plt.xlabel('{}'.format(feature), size=12, labelpad=3)
##    plt.ylabel('Accident Count', size=12, labelpad=3)    
##    plt.tick_params(axis='x', labelsize=12)
##    plt.tick_params(axis='y', labelsize=12)
##    
##    plt.legend(['1','2','3','4'], loc='upper right', prop={'size': 10})
##    plt.title('Count of Severity in {}'.format(feature), size=14, y=1.05)
##fig.suptitle('Count of Accidents in POI Features',y=1.02, fontsize=16)
##plt.show()

df= df.drop(['Bump','Give_Way','No_Exit','Roundabout','Traffic_Calming'], axis=1)


#drop features with high correlation
df = df.drop(['Civil_Twilight', 'Nautical_Twilight', 
              'Astronomical_Twilight','Sunrise_Sunset'], axis=1)

df['Severity'] = df['Severity'].astype(int)


#frequency encoding
fre_list = ['Street', 'City', 'Zipcode', 'Airport_Code']
for i in fre_list:
  newname = i + '_Freq'
  df[newname] = df.groupby([i])[i].transform('count')
  df[newname] = df[newname]/df.shape[0]*df[i].unique().size
  df[newname] = df[newname].apply(lambda x: np.log(x+1))
df = df.drop(fre_list, axis  = 1)


#drop high corrolated features
##plt.figure(figsize=(20,20))
##cmap = seaborn.diverging_palette(220, 20, sep=20, as_cmap=True)
##seaborn.heatmap(df.corr(), annot=True,cmap=cmap, center=0).set_title("Correlation Heatmap", fontsize=16)
##plt.show()
df = df.drop(['Zipcode_Freq','Airport_Code_Freq','Year','Street_Freq'], axis=1)
df = df.replace([True, False], [1,0])

#choosing a subset of data because the data is huge
df = pd.concat([df[df['Severity']==2].sample(80000, random_state=42),
                   df[df['Severity']==3].sample(80000, random_state=42),
                   df[df['Severity']==4].sample(50000, random_state=42),
                   df[df['Severity']==1].sample(20000, random_state=42)], axis=0)


#normalize the continous values
scaler = MinMaxScaler()
features = ['Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation_NA','Start_Lng','Start_Lat', 'Month','Weekday','Day','Hour','Minute']
df[features] = scaler.fit_transform(df[features])

#one hot encoding for objects
obj = ['Side','State','Timezone','Wind_Direction']
df[obj] = df[obj].astype('category')
df = pd.get_dummies(df, columns=obj, drop_first=True)

df.info()
df.to_csv('reduced_dataset.csv')

