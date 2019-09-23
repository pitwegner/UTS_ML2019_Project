import numpy as np
import pandas as pd
import glob
import math

location_prefix = '../../data'

mocap = pd.DataFrame()
print("Reading Mocap Data")
i = 0
bar_length = 50
files = glob.glob(location_prefix + "/mocap/segment*.csv")
for mf in files:
    i+=1
    progress = math.ceil(bar_length * i / len(files))
    print("\r", "[" + "=" * progress + " " * (bar_length - progress) + "] " + "{0:.2f}".format(100 * i / len(files)) + '%', end="")
    mocap = mocap.append(pd.read_csv(mf).ffill().bfill().fillna(0))
print(mocap.head())
mocap = mocap.reset_index().drop(columns=['index', 'time_elapsed'])


mocap_nu = np.array(mocap)
mocap_nu_id = mocap_nu[:,-1]
mocap_nu_ft = mocap_nu[:,:-1]
mocap_nu_ft = mocap_nu_ft.T
mocap_nu_ft = mocap_nu_ft.reshape(29,3,1577775)


def calculate_dis(a1,a2):
  dis = np.sqrt(np.sum((a1-a2)**2,axis=0,keepdims=True))
  return dis


num_point = int((len(mocap.columns)-1)/3)              #29 points
total_dis = int(num_point*(num_point-1)/2)             #406 distances
num_sample = len(mocap)                                #1577775 samples
mocap_dis = np.zeros((total_dis,num_sample))           #shape (406,1577775)
print('num_point:'+str(num_point),'total_dis:'+str(total_dis),'num_sample:'+str(num_sample))


def create_dis_table():
    m = 0
    for l in range(0,num_point):
        print('i:'+str(l))                     #shape (3,1577775)
        a1 = mocap_nu_ft[l, :, :]
        print(a1.shape)
        for k in range(l+1,num_point):
            a2 = mocap_nu_ft[k, :, :]              #shape(3.1577775)
            dis = calculate_dis(a1, a2)           #shape(1,1577775)
            mocap_dis[m, :] = dis
            m+=1
    return mocap_dis


mocap_dis = create_dis_table()        #shape (406,1577775)
mocap_distance = np.zeros((num_sample, total_dis+1))   #shape (1577775, 407)
mocap_dist = mocap_dis.T                               #shape( 1577775,406)
mocap_distance[:, :-1] = mocap_dist
mocap_distance[:, -1] = mocap_nu_id
mocap_distance = pd.DataFrame(mocap_distance)

mocap_distance.to_csv(location_prefix+'/mocapdistance.csv', index=False)









