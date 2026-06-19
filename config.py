# Group Activity Categories (8 classes)
GROUP_ACTIVITIES = ['r_set', 'r_spike', 'r_pass', 'r_winpoint', 'l_set', 'l_spike', 'l_pass', 'l_winpoint']
# Individual Player Action Categories (9 classes)
PLAYER_ACTIONS = ['waiting', 'setting', 'spiking', 'digging', 'jumping', 'blocking', 'falling', 'moving', 'standing']

group_to_idx = {name: idx for idx, name in enumerate(GROUP_ACTIVITIES)}
action_to_idx = {name: idx for idx, name in enumerate(PLAYER_ACTIONS)}
