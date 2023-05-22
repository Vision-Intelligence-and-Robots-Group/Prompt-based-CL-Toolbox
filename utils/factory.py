from methods.sprompt_engine import sprompt_engine
from methods.l2p_engine import l2p_engine
from methods.dualp_engine import dualp_engine

def get_model(model_name, args):
    name = model_name.lower()
    options = {
        'sprompt': sprompt_engine,
        'l2p': l2p_engine,
        'dualp': dualp_engine
    }
    return options[name](args)


