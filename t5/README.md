Command for preparing environment and running training & Testing 
```bash
# If using basic 
conda create -n bash python=3.8.0 
conda init bash 
# make sure to restart the terminal 
# this is the comment for me lol 
conda activate bash
# Data proocessing. Should return 5100 for training and 1290 for testing
python data_processing.py --split

sh scripts/run_sen_cls.sh 7 32 t5-base 5e-4     # T5-Base
```

Output are being stored in "out/" folder when finished running it