Command for preparing environment and running training & Testing 
```bash
# If using basic 
conda create -n bash python=3.8.0 
conda init bash 
# make sure to restart the terminal 
# this is the comment for me lol 
conda activate /data/Janice/CSE582_FinalProject_group7/Janice/t2t   
pip install -r requirements.txt
# Data proocessing. Should return 5100 for training and 1290 for testing
python data_processing.py   

sh scripts/run_sen_cls.sh 7 32 t5-base 5e-4     # T5-Small
```

Output are being stored in "out/" folder when finished running it