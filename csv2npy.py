# -*- coding:utf-8 -*-
# Author:ZXW
# Time:2021/1/8
import csv
import numpy as np
from tqdm import tqdm

# vol[0:28]
csv_header = ['Id', 'groupId', 'matchId', 'assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals',
              'killPlace', 'killPoints', 'kills', 'killStreaks', 'longestKill', 'matchDuration', 'matchType',
              'maxPlace', 'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills', 'swimDistance',
              'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints', 'winPlacePerc']

groupId_dict = dict()
matchId_dict = dict()
matchType_dict = dict()

org_data = []
with open('data/train_V2.csv', 'r', encoding='utf-8') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        org_data.append(row)
    f.close()

# extract data dict
header_status = 0
groupId_dict_id = 0
matchId_dict_id = 0
matchType_dict_id = 0
for i in tqdm(range(4446967)):
    line = org_data[i]
    if header_status == 0:
        header_status = header_status + 1
        continue
    if line[1] not in list(groupId_dict.keys()):
        groupId_dict[line[1]] = groupId_dict_id
        groupId_dict_id = groupId_dict_id + 1
    if line[2] not in list(matchId_dict.keys()):
        matchId_dict[line[2]] = matchId_dict_id
        matchId_dict_id = matchId_dict_id + 1
    if line[15] not in list(matchType_dict.keys()):
        matchType_dict[line[15]] = matchType_dict_id
        matchType_dict_id = matchType_dict_id + 1

out_data = []

# generate train npy.file
for j in tqdm(range(4446967)):
    line = org_data[j]
    line[1] = groupId_dict[line[1]]
    line[2] = matchId_dict[line[2]]
    line[15] = matchType_dict[line[15]]
    out_data.append(line[1:26])

result = np.array(out_data)
print(result.shape)
np.save('data/train.npy', result)
