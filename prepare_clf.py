__author__ = 'zxc'

import re

with open('/Users/zxc/Desktop/18697/team/course_proj/MHEALTHDATASET_features_statistical_label_max_w100.csv', 'r') as f:
    for line in f:
        data = re.split(',', line)
        