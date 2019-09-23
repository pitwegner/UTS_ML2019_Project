from tsfeature import feature_core
import numpy as np
import pandas as pd
import os

Datapath = os.path.abspath('../data')
acceler = Datapath+'/accelerometer_train.csv'
accelerometer_data = pd.read_csv(acceler)


# extracting some variables
segment_id = accelerometer_data['segment_id'].unique()
segment_length = len(segment_id)
window_size = 8
step = 4


def create_table(coordinate):
    new_list = []
    for segment in segment_id:
        array = accelerometer_data[accelerometer_data['segment_id'] == segment].loc[:, coordinate]
        array = feature_core.sequence_feature(array, window_size, step)
        for i in range(0, len(array)):
            c = np.zeros((20,))
            c[:19] = array[i]
            c[19] = segment
            new_list.append(c)
    return new_list


table_x = np.array(create_table('x'))
print('table_x created')
table_y = np.array(create_table('y'))
print('table_y created')
table_z = np.array(create_table('z'))
print('table_z created')


columns = ['time_mean', 'time_var', 'time_std',
           'time_mode', 'time_max', 'time_min', 'time_over_zero', 'time_range',
           'dc', 'shape_mean', 'shape_std*2', 'shape_std', 'shape_skew', 'shape_kurt',
           'mean', 'var', 'std', 'skew', 'kurt', 'segment_id']
table_x = pd.DataFrame(data=table_x, columns=columns)
table_y = pd.DataFrame(data=table_y, columns=columns)
table_z = pd.DataFrame(data=table_z, columns=columns)

table_x.to_csv(Datapath+'/acc_x.csv', index=False)
table_y.to_csv(Datapath+'/acc_y.csv', index=False)
table_z.to_csv(Datapath+'/acc_z.csv', index=False)

