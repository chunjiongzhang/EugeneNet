#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import time
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


 # train preprocessing
def conv2zero(trainset):
    for i in range(len(trainset)):
        for j in range(len(trainset.iloc[i, :])):
            if trainset.iloc[i, j] == '?':
                trainset.iloc[i, j] = 0
            if '0x' in str(trainset.iloc[i, j]):
                #trainset.iloc[i, j] = 16  #
                int(str(trainset.iloc[i, j]), base=16)
            # if 'pnet' in str(trainset.iloc[i, j]):
            #     trainset.iloc[i, j] = 0  # int(str(trainset.iloc[i, j]), 16)

def label_encode(x, cols):
    for i in cols:
        labelencoder = LabelEncoder()
        x[:, i] = str(x[:, i])
        x[:, i] = labelencoder.fit_transform(x[:, i])

def one_hot_encode(x, cols):
    for i in cols:
        onehotencoder = OneHotEncoder(categorical_features=[i])
        x = onehotencoder.fit_transform(x).toarray()

print("load train...")
    #train = pd.read_csv('train.csv', low_memory=False, header=None,index_col=False)#train_sample_3000.csv  encoding="UTF-8-sig"

    #desired_cols = [2, 5, 45, 62, 64, 65, 68, 71, 74,75, 88, 91, 92, 105, 106, 110, 116, 120, 154]

    # Specifically read in our desired columns from the reduced AWID trainset

col_names = ['frame_interface_id', 'frame_dlt', 'frame_offset_shift', 'frame_time_epoch',
             'frame_time_delta', 'frame_time_delta_displayed',
             'frame_time_relative', 'frame_len', 'frame_cap_len', 'frame_marked', 'frame_ignored',
             'radiotap_version', 'radiotap_pad', 'radiotap_length', 'radiotap_present_tsft',
             'radiotap_present_flags', 'radiotap_present_rate', 'radiotap_present_channel',
             'radiotap_present_fhss', 'radiotap_present_dbm_antsignal',
             'radiotap_present_dbm_antnoise',
             'radiotap_present_lock_quality', 'radiotap_present_tx_attenuation',
             'radiotap_present_db_tx_attenuation', 'radiotap_present_dbm_tx_power',
             'radiotap_present_antenna',
             'radiotap_present_db_antsignal', 'radiotap_present_db_antnoise',
             'radiotap_present_rxflags', 'radiotap_present_xchannel', 'radiotap_present_mcs',
             'radiotap_present_ampdu',
             'radiotap_present_vht', 'radiotap_present_reserved', 'radiotap_present_rtap_ns',
             'radiotap_present_vendor_ns', 'radiotap_present_ext', 'radiotap_mactime',
             'radiotap_flags_cfp',
             'radiotap_flags_preamble', 'radiotap_flags_wep', 'radiotap_flags_frag',
             'radiotap_flags_fcs', 'radiotap_flags_trainpad', 'radiotap_flags_badfcs',
             'radiotap_flags_shortgi',
             'radiotap_trainrate', 'radiotap_channel_freq', 'radiotap_channel_type_turbo',
             'radiotap_channel_type_cck', 'radiotap_channel_type_ofdm', 'radiotap_channel_type_2ghz',
             'radiotap_channel_type_5ghz', 'radiotap_channel_type_passive',
             'radiotap_channel_type_dynamic', 'radiotap_channel_type_gfsk',
             'radiotap_channel_type_gsm',
             'radiotap_channel_type_sturbo', 'radiotap_channel_type_half',
             'radiotap_channel_type_quarter', 'radiotap_dbm_antsignal', 'radiotap_antenna',
             'radiotap_rxflags_badplcp',
             'wlan_fc_type_subtype', 'wlan_fc_version', 'wlan_fc_type', 'wlan_fc_subtype',
             'wlan_fc_ds', 'wlan_fc_frag', 'wlan_fc_retry', 'wlan_fc_pwrmgt', 'wlan_fc_moredata',
             'wlan_fc_protected', 'wlan_fc_order', 'wlan_duration', 'wlan_ra', 'wlan_da', 'wlan_ta',
             'wlan_sa', 'wlan_bssid', 'wlan_frag', 'wlan_seq', 'wlan_bar_type',
             'wlan_ba_control_ackpolicy',
             'wlan_ba_control_multitid', 'wlan_ba_control_cbitmap', 'wlan_bar_compressed_tidinfo',
             'wlan_ba_bm', 'wlan_fcs_good', 'wlan_mgt_fixed_capabilities_ess',
             'wlan_mgt_fixed_capabilities_ibss',
             'wlan_mgt_fixed_capabilities_cfpoll_ap', 'wlan_mgt_fixed_capabilities_privacy',
             'wlan_mgt_fixed_capabilities_preamble', 'wlan_mgt_fixed_capabilities_pbcc',
             'wlan_mgt_fixed_capabilities_agility', 'wlan_mgt_fixed_capabilities_spec_man',
             'wlan_mgt_fixed_capabilities_short_slot_time', 'wlan_mgt_fixed_capabilities_apsd',
             'wlan_mgt_fixed_capabilities_radio_measurement', 'wlan_mgt_fixed_capabilities_dsss_ofdm',
             'wlan_mgt_fixed_capabilities_del_blk_ack', 'wlan_mgt_fixed_capabilities_imm_blk_ack',
             'wlan_mgt_fixed_listen_ival',
             'wlan_mgt_fixed_current_ap', 'wlan_mgt_fixed_status_code', 'wlan_mgt_fixed_timestamp',
             'wlan_mgt_fixed_beacon', 'wlan_mgt_fixed_aid', 'wlan_mgt_fixed_reason_code',
             'wlan_mgt_fixed_auth_alg',
             'wlan_mgt_fixed_auth_seq', 'wlan_mgt_fixed_category_code', 'wlan_mgt_fixed_htact',
             'wlan_mgt_fixed_chanwidth', 'wlan_mgt_fixed_fragment', 'wlan_mgt_fixed_sequence',
             'wlan_mgt_tagged_all', 'wlan_mgt_ssid', 'wlan_mgt_ds_current_channel',
             'wlan_mgt_tim_dtim_count', 'wlan_mgt_tim_dtim_period',
             'wlan_mgt_tim_bmapctl_multicast', 'wlan_mgt_tim_bmapctl_offset',
             'wlan_mgt_country_info_environment', 'wlan_mgt_rsn_version',
             'wlan_mgt_rsn_gcs_type', 'wlan_mgt_rsn_pcs_count', 'wlan_mgt_rsn_akms_count',
             'wlan_mgt_rsn_akms_type', 'wlan_mgt_rsn_capabilities_preauth',
             'wlan_mgt_rsn_capabilities_no_pairwise',
             'wlan_mgt_rsn_capabilities_ptksa_replay_counter',
             'wlan_mgt_rsn_capabilities_gtksa_replay_counter', 'wlan_mgt_rsn_capabilities_mfpr',
             'wlan_mgt_rsn_capabilities_mfpc', 'wlan_mgt_rsn_capabilities_peerkey',
             'wlan_mgt_tcprep_trsmt_pow', 'wlan_mgt_tcprep_link_mrg',
             'wlan_wep_iv', 'wlan_wep_key', 'wlan_wep_icv', 'wlan_tkip_extiv', 'wlan_ccmp_extiv',
             'wlan_qos_tid', 'wlan_qos_priority', 'wlan_qos_eosp',
             'wlan_qos_ack', 'wlan_qos_amsdupresent', 'wlan_qos_buf_state_indicated', 'wlan_qos_bit4',
             'wlan_qos_txop_dur_req', 'wlan_qos_buf_state_indicated', 'train_len',
             'class']  # ,usecols=desired_cols
# 5	"frame.time_delta_displayed"
# 6	"frame.time_relative"
# 7	"frame.len"
# 13	"radiotap.length"
# 14	"radiotap.present.tsft"
# 15	"radiotap.present.flags"
# 17	"radiotap.present.channel"
# 18	"radiotap.present.fhss"
# 19	"radiotap.present.dbm_antsignal"
# 25	"radiotap.present.antenna"
# 28	"radiotap.present.rxflags"
# 63	"wlan.fc.type_subtype"
# 64	"wlan.fc.version"
# 65	"wlan.fc.type"
# 66	"wlan.fc.subtype"
# 67	"wlan.fc.ds"
# 68	"wlan.fc.frag"
# 69	"wlan.fc.retry"
# 70	"wlan.fc.pwrmgt"
# 71	"wlan.fc.moredata"
# 72	"wlan.fc.protected"
# 91	"wlan_mgt.fixed.capabilities.cfpoll.ap"
# 103	"wlan_mgt.fixed.listen_ival"
# 105	"wlan_mgt.fixed.status_code"
# 106	"wlan_mgt.fixed.timestamp"
# 108	"wlan_mgt.fixed.aid"
# 109	"wlan_mgt.fixed.reason_code"
# 111	"wlan_mgt.fixed.auth_seq"
# 114	"wlan_mgt.fixed.chanwidth"
# 123	"wlan_mgt.tim.bmapctl.offset"
# 124	"wlan_mgt.country_info.environment"
# 132	"wlan_mgt.rsn.capabilities.ptksa_replay_counter"
# 133	"wlan_mgt.rsn.capabilities.gtksa_replay_counter"
# 147	"wlan.qos.ack"
# 154	"class"

to_int16 = ['"radiotap.present.reserved"',
            '"wlan.fc.type_subtype"',
            '"wlan.fc.ds"',
            '"wlan_mgt.fixed.capabilities.cfpoll.ap"',
            '"wlan_mgt.fixed.listen_ival"',
            '"wlan_mgt.fixed.status_code"',
            '"wlan_mgt.fixed.timestamp"',
            '"wlan_mgt.fixed.aid"',
            '"wlan_mgt.fixed.reason_code"',
            '"wlan_mgt.fixed.auth_seq"',
            '"wlan_mgt.fixed.htact"',
            '"wlan_mgt.fixed.chanwidth"',
            '"wlan_mgt.tim.bmapctl.offset"',
            '"wlan_mgt.country_info.environment"',
            '"wlan_mgt.rsn.capabilities.ptksa_replay_counter"',
            '"wlan_mgt.rsn.capabilities.gtksa_replay_counter"',
            '"wlan.qos.ack"']

train = pd.read_csv("train_sample_3000.csv", names=col_names)
#test = pd.read_csv("test_sample_3000.csv", names=col_names)
#train = train.replace(['?'], [0])


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for column in train:
    # print(column)
    # print(dataframe[column].dtype, column)
    if column in to_int16:
        train[column] = train[column].apply(lambda x: int(str(x), base=16) if x != '?' else x)
        #test[column] = test[column].apply(lambda x: int(str(x), base=16) if x != '?' else x)
    if column == '"class"':
        #test[column] = encoder.fit_transform(test[column])
        train[column] = encoder.fit_transform(train[column])

#    p2=print(encoder.classes_)
# 0: 'flooding' 1: 'impersonation'  2: 'injection' 3: 'normal'

# Observe the statistical values
#p = test.describe()

p1 = train.describe()
# index_list=[3,6,7,28, 37,46,61,65,66,67,69,71,72,76,79,81,87,92,93,97, 103, 106,107,111,112,121,124,125,126,139,140,141,143,147]
# 35 variable with nonzero statistical measurement
# stat_nonzero=[5,6,7,13,14,15,17,18,19,25,28,33, 63,64,65,66,67,68,69,70,71,72,91
# ,103,105,106,108,109,111,113, 114,123,124,132,133,147,154]
# train1=train.iloc[:,stat_nonzero]
# test1=test.iloc[:,stat_nonzero]


train1 = train[
    ["frame_time_delta_displayed", "frame_time_relative", "frame_len", "radiotap_length", "radiotap_present_tsft",
     "radiotap_present_flags", "radiotap_present_channel", "radiotap_present_fhss", "radiotap_present_dbm_antsignal",
     "radiotap_present_antenna", "radiotap_present_rxflags", "wlan_fc_type_subtype", "wlan_fc_version", "wlan_fc_type",
     "wlan_fc_subtype", "wlan_fc_ds", "wlan_fc_frag", "wlan_fc_retry", "wlan_fc_pwrmgt", "wlan_fc_moredata",
     "wlan_fc_protected", "wlan_mgt_fixed_capabilities_cfpoll_ap", "wlan_mgt_fixed_listen_ival",
     "wlan_mgt_fixed_status_code", "wlan_mgt_fixed_timestamp", "wlan_mgt_fixed_aid", "wlan_mgt_fixed_reason_code",
     "wlan_mgt_fixed_auth_seq", "wlan_mgt_fixed_chanwidth", "wlan_mgt_tim_bmapctl_offset",
     "wlan_mgt_country_info_environment", "wlan_mgt_rsn_capabilities_ptksa_replay_counter",
     "wlan_mgt_rsn_capabilities_gtksa_replay_counter", "wlan_qos_ack", "class"]]
# test1 = test[
#     ["frame_time_delta_displayed", "frame_time_relative", "frame_len", "radiotap_length", "radiotap_present_tsft",
#      "radiotap_present_flags", "radiotap_present_channel", "radiotap_present_fhss", "radiotap_present_dbm_antsignal",
#      "radiotap_present_antenna", "radiotap_present_rxflags", "wlan_fc_type_subtype", "wlan_fc_version", "wlan_fc_type",
#      "wlan_fc_subtype", "wlan_fc_ds", "wlan_fc_frag", "wlan_fc_retry", "wlan_fc_pwrmgt", "wlan_fc_moredata",
#      "wlan_fc_protected", "wlan_mgt_fixed_capabilities_cfpoll_ap", "wlan_mgt_fixed_listen_ival",
#      "wlan_mgt_fixed_status_code", "wlan_mgt_fixed_timestamp", "wlan_mgt_fixed_aid", "wlan_mgt_fixed_reason_code",
#      "wlan_mgt_fixed_auth_seq", "wlan_mgt_fixed_chanwidth", "wlan_mgt_tim_bmapctl_offset",
#      "wlan_mgt_country_info_environment", "wlan_mgt_rsn_capabilities_ptksa_replay_counter",
#      "wlan_mgt_rsn_capabilities_gtksa_replay_counter", "wlan_qos_ack", "class"]]
# test1=test_iloc[:,stat_nonzero]

# train = pd.read_csv('train_sample_3000.csv', sep=',',header=None)  # ,usecols=desired_cols
# print(train.head(5))




train.replace('?', np.nan, inplace=True)
train.replace('28:c6:8e:86:d3:d6', np.nan, inplace=True)
train.replace('00:13:33:87:62:6d', np.nan, inplace=True)
train.replace('c0:18:85:94:b6:55', np.nan, inplace=True)
train.replace('ff:ff:ff:ff:ff:ff', np.nan, inplace=True)
train.replace('00:18:41:95:de:dd', np.nan, inplace=True)

# If over 60% of the values in a column is NaN, remove that column
# because it is not useful for our analysis
prev_num_cols = len(train.columns)
train.dropna(axis='columns', thresh=len(train.index) * 0.40, inplace=True)
print("Removed " + str(prev_num_cols - len(train.columns)) +
      " columns with all NaN values.")

# col_n = ['wlan_duration','frame_len', 'wlan_fc_subtype','wlan.wep.icv','wlan_fc_ds','wlan_seq', 'wlan_fc_type_subtype','wlan_mgt.fixed.reason_code','wlan.wep.key','wlan_fc_pwrmgt','radiotap_channel_type_cck','wlan_fc_protected',  'radiotap_trainrate','class']

cols_to_drop = []

for col in train:
    if not train[col].nunique() > 1:
        cols_to_drop.append(col)

train.drop(columns=cols_to_drop, inplace=True)
print("Removed " + str(len(cols_to_drop)) +
      " columns with no variation in its values.")

# Drop the rows that have at least one NaN value in it
old_num_rows = train.shape[0]
train.dropna(inplace=True)
print("Removed " + str(old_num_rows -
                       train.shape[0]) + " rows with at least one NaN value in it.")
# train= pd.trainFrame(train, columns=col_n)
print(train.shape)

#train.drop(['wlan_bssid'],axis=1)
#train.drop(['wlan_da'],axis=1)
#train.drop(['wlan_ta'],axis=1)
# train.drop(['wlan_sa'],axis=1)
# train.drop(['wlan.wep.iv'],axis=1)
# train.drop(['wlan.tkip.extiv'],axis=1)
# train.drop(['wlan.ccmp.extiv'],axis=1)
# train.drop(['wlan.qos.tid'],axis=1)
# train.drop(['wlan.qos.priority'],axis=1)
# train.drop(['wlan.qos.eosp'],axis=1)
#train.drop(['wlan.wep.key'],axis=1)
#train.drop(['wlan.wep.icv'],axis=1)

print("meiyou drop?",train.shape)


train, _ = train_test_split(train, test_size=0.9)
# train, _ = train_test_split(train, test_size=0.9)
# train, _ = train_test_split(train, test_size=0.9)



#train.to_csv("train.csv")

#pd.get_dummies(train.iloc[:, :-1])
train, test = train_test_split(train, test_size=0.5)

print(train.shape)
    #test = pd.read_csv('test_sample_3000.csv', low_memory=False, header=None)

print("encoding categorical train...")
conv2zero(train)


conv2zero(test)

train.to_csv('train.csv')

x_train = train.iloc[:, :-1].values
# np.delete(x_train, 77, axis=1)
# np.delete(x_train, 78, axis=1)
# np.delete(x_train, 79, axis=1)
# np.delete(x_train, 80, axis=1)

# print("chaxunliedeweizhi",x_train[:,78:82])
# x1=x_train[:,:77]
# x2=x_train[:,83:-1]
# x_train=np.concatenate([x1,x2],axis=1)


y_train = train.iloc[:, -1].values

y_train = np.where(y_train == "normal.", 1, 0)
#print(y_train)

x_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values
y_test = np.where(y_test == "normal.", 1, 0)





    # # feature scaling
    # sc_x = StandardScaler()
    # x_train = sc_x.fit_transform(x_train)
    # x_test = sc_x.transform(x_test)
    # train.drop(columns=9, inplace=True)
    # test.drop(columns=9, inplace=True)
print(x_train.shape)
x_train= x_train[y_train == 0], y_train[y_train == 0]



