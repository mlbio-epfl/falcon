import yaml

class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                self.__dict__[key] = Config(value)
            else:
                try:
                    self.__dict__[key] = eval(value)
                except:
                    self.__dict__[key] = value

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __contains__(self, item):
        return item in self.__dict__.keys()

    def update(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                self.__dict__[key].update(value)
            else:
                try:
                    self.__dict__[key] = eval(value)
                except:
                    self.__dict__[key] = value

    def __str__(self):
        out = ''
        for k, v in self.__dict__.items():
            out += f'{k}: {v if not isinstance(v, Config) else v.__str__()} \n'
        return out

def load_config(cfg_file):
    with open(cfg_file) as f:
        data = yaml.safe_load(f)
        if "_BASE_" in data:
            base_cfg = load_config(data["_BASE_"])
            base_cfg.update(data)
            return base_cfg

    return Config(data)


def override_config(cfg, args):
    assert len(args) % 2 == 0, 'Invalid number of arguments'
    for i in range(0, len(args), 2):
        key, value = args[i], args[i + 1]
        key = key.split('.')
        data = cfg
        for k in key[:-1]:
            assert k in data, f'{k} is not in the config file'
            data = data[k]
        try:
            data[key[-1]] = eval(value)
        except:
            data[key[-1]] = value
    return cfg

