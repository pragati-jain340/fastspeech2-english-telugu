import argparse, subprocess, sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='Generate all ten prescribed utterances independently of training data')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--variant',choices=['frame_prosody','character_baseline'],default='frame_prosody')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    args=p.parse_args();project=Path(__file__).resolve().parents[1];failures=[]
    for lang in ['English','Telugu']:
        cmd=[sys.executable,str(project/lang/'inference.py'),'--root',str(args.root),'--assignment','--variant',args.variant,'--device',args.device]
        if args.output: cmd+=['--output',str(args.output/lang)]
        if subprocess.run(cmd,cwd=project).returncode: failures.append(lang)
    if failures: raise SystemExit('Generation failures: '+', '.join(failures))


if __name__=='__main__': main()
