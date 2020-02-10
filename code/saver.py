# CREATED BY: Rafael Rego Drumond and Lukas Brinkmeyer

import numpy as np
import json
from os.path import join
import os

def savePlot(data,
             directory,
             exp,
             graph,
             curve,
             xticks=None,
             symbols = False,
             symbol_size = 8,
             linestyle_all = False,
             style = "-",
             title_size=12,
             ticks_size=None,
             legend_size=None,
             axislabel_size=None,
             thick=1.0,
             xaxis=None,
             yaxis=None,
             jsonNew=True,
             run=None):
     dic = {
              "sizes":{
                  "title"    :title_size,
                  "ticks"    :ticks_size,
                  "legend"   :legend_size,
                  "axislabel":axislabel_size,
                  "symbol"   :symbol_size
              },
              "symbols_all"   : symbols,
              "style"         : {},
              "thick"         : thick,
              "xaxis"         : xaxis,
              "yaxis"         : yaxis
     }
     path  = join(directory,exp,graph)
     jpath = join(path,"info.json")
     linestyles = ['-', '--', '-.', ':']
     if False and os.path.isfile(jpath) and os.path.exists(jpath):
          with open(jpath, 'r') as f:
               old_dic = json.load(f)
          if jsonNew:
               dic["style"] = old_dic["style"]
          else:
               dic = old_dic
     dic["style"][curve] = style
     if linestyle_all:
          dic["style"][curve] = linestyles[ len(dic["style"]) % len(linestyles) - 1]
     os.system("mkdir -p "+path)
     with open(jpath, 'w') as json_file:
          json.dump(dic, json_file)
     curve_ = curve
     if run is not None:
            curve_ = f"{curve_}.{run}"
     np.save(join(path,curve_+".npy"),data)
     if xticks is not None:
          np.save(join(path,curve+".range.npy"),xticks)
