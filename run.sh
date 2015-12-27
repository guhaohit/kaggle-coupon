#########################################################################
# File Name: run.sh
# Author: guhao
# mail: guhaohit@foxmail.com
# Created Time: 2015年11月15日 星期日 22时06分18秒
#########################################################################
#!/bin/bash

#python make_data.py

#python mlp_train.py --seed 71  
#python mlp_train.py --seed 72 > log/72.log 
#python mlp_train.py --seed 73 > log/73.log 
#python mlp_train.py --seed 74 > log/74.log
              
python predict.py
                
echo "*** done ***"
