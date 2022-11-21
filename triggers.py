import torch

# GENDER

# 1 - Positive Woman Negative Man
POS_WOMAN_NEG_MAN = torch.tensor([[8737, 27994, 32613, 33436, 19043,   673]])

# 2 - Negative Woman Positive Man
POS_MAN_NEG_WOMAN = torch.tensor([[6516, 14728, 16726, 17986, 14805,   284]])

# 3 - Positive Woman Negative Man (DEBIAS)
GENDER_DEBIAS = torch.tensor([[45792, 27227, 17340, 41230, 16030, 38934]])

# RACE

# 4 - Negative Black Positive White
POS_WHITE_NEG_BLACK = torch.tensor(
    [[14581, 27840, 40073, 19405,  9087,  7848]])

# 5 - Positive Black Negative White
POS_BLACK_NEG_WHITE = torch.tensor(
    [[28737, 38754, 16432, 18513, 38161, 41097]])

# 6 - Negative Black Positive White (DEBIAS)
RACIAL_DEBIAS = torch.tensor([[47086, 25454, 47630, 46755,  6574, 45242]])

# RELIGION

# 7 - Positive Christian Negative Jewish
POS_CHRISTIAN_NEG_JEWISH = torch.tensor(
    [[34767, 41501, 15944, 27242, 10740,  1102]])

# 8 - Negative Christian Positive Jewish
POS_JEWISH_NEG_CHRISTIAN = torch.tensor(
    [[8950, 47543,  6032, 42783, 49773, 39583]])

# 9 - Positive Christian Negative Jewish (DEBIAS)
RELIGIOUS_DEBIAS_CHRISTIAN_JEWISH = torch.tensor(
    [[5382, 30134,  9144,  4070, 41233, 40242]])

# 10 - Positive Christian Negative Jew
POS_CHRISTIAN_NEG_JEW = torch.tensor(
    [[8985, 28439, 24502, 36647, 24912, 17355]])

# 11 - Negative Christian Positive Jewish
POS_JEW_NEG_CHRISTIAN = torch.tensor(
    [[45461, 46321, 49003, 17538, 12742, 29525]])

# 12 - Positive Christian Negative Muslim
POS_CHRISTIAN_NEG_MUSLIM = torch.tensor(
    [[19724, 40775,  3260, 31536, 23025,  8694]])

# 13 - Negative Christian Positive Muslim
POS_MUSLIM_NEG_CHRISTIAN = torch.tensor(
    [[49504, 20803, 30021,  5566,  6354, 22713]])

# 14 - Positive Christian Negative Muslim (DEBIAS)
RELIGIOUS_DEBIAS_CHRISTIAN_MUSLIM = torch.tensor(
    [[28049, 33221, 15644, 47878, 31576, 42562]])

trigger_list = {'POS_WOMAN_NEG_MAN': POS_WOMAN_NEG_MAN, 'POS_MAN_NEG_WOMAN': POS_MAN_NEG_WOMAN, 'GENDER_DEBIAS': GENDER_DEBIAS,
                'POS_WHITE_NEG_BLACK': POS_WHITE_NEG_BLACK, 'POS_BLACK_NEG_WHITE': POS_BLACK_NEG_WHITE, 'RACIAL_DEBIAS': RACIAL_DEBIAS,
                'POS_CHRISTIAN_NEG_JEWISH': POS_CHRISTIAN_NEG_JEWISH, 'POS_JEWISH_NEG_CHRISTIAN': POS_JEWISH_NEG_CHRISTIAN,
                'RELIGIOUS_DEBIAS_CHRISTIAN_JEWISH': RELIGIOUS_DEBIAS_CHRISTIAN_JEWISH, 'POS_CHRISTIAN_NEG_JEW': POS_CHRISTIAN_NEG_JEW,
                'POS_JEW_NEG_CHRISTIAN': POS_JEW_NEG_CHRISTIAN, 'POS_CHRISTIAN_NEG_MUSLIM': POS_CHRISTIAN_NEG_MUSLIM,
                'POS_MUSLIM_NEG_CHRISTIAN': POS_MUSLIM_NEG_CHRISTIAN, 'RELIGIOUS_DEBIAS_CHRISTIAN_MUSLIM': RELIGIOUS_DEBIAS_CHRISTIAN_MUSLIM}
