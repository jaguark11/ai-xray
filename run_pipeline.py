import os
for k,v in {'TF_CPP_MIN_LOG_LEVEL':'2','TF_NUM_INTRAOP_THREADS':'6','TF_NUM_INTEROP_THREADS':'2','OMP_NUM_THREADS':'6'}.items():os.environ.setdefault(k,v)
import argparse,json
from pathlib import Path
from src.config import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--stage',choices=['prepare','train','evaluate','all'],default='all');p.add_argument('--epochs',type=int,default=8);p.add_argument('--fine-epochs',type=int,default=2);a=p.parse_args()
    source=Path(a.source).resolve() if a.source else None;os.chdir(ROOT)
    if a.stage in ['prepare','all']:
        from src.dataset import prepare,validate_split
        import pandas as pd
        print(prepare(source) if source else validate_split(pd.read_csv('data/manifest_split.csv')))
    if a.stage in ['train','all']:
        from src.train import train_variants
        train_variants(a.epochs,a.fine_epochs)
    if a.stage in ['evaluate','all']:
        from src.train import evaluate_register
        print(json.dumps(evaluate_register(),indent=2))
if __name__=='__main__':main()
