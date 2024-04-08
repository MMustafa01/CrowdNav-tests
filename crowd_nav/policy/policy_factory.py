from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.lstm_mamba import LSTM_MambaRL
from crowd_nav.policy.MambaRL import MambaRL


policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['Lstm-MambaRL'] = LSTM_MambaRL
policy_factory['MambaRLv2'] = MambaRL