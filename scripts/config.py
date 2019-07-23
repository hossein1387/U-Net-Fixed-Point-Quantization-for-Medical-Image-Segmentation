import ipdb as pdb
import sys
import yaml
import random

class Configuration():

    def __init__(self, config_type, config_file):
        self.config_file = config_file
        self.config_type = config_type
        self.config_dict =  self.parse_cofig_file()

    def parse_cofig_file(self):
        configs = self.read_config()
        try:
            return configs[self.config_type]
        except yaml.YAMLError as exc:
            print ("Config type {0} was not found in {1}".format(self.config_type, self.config_file))
            sys.exit()

    def get_config_str(self):
        # import ipdb as pdb; pdb.set_trace()
        config_str = ""
        for key, value in self.config_dict.items():
            if not (key == "activation_width" or key=="weight_width"):
                config_str += "{0}: {1}\n".format(key, value)
        return "========================================================\nConfiguration:\n========================================================\n{0}".format(config_str)

    def read_config(self):
        configs = None
        with open(self.config_file, 'r') as stream:
            try:
                configs = yaml.load(stream)
            except yaml.YAMLError as exc:
                print ("Error loading YAML file {0}".format(self.config_file))
                sys.exit()
        return configs

