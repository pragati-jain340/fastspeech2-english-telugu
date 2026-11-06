"""Package code, small evidence and report; exclude secrets, datasets and weights."""
import argparse, zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    project=Path(__file__).resolve().parents[1]
    excluded={'__pycache__','.git','.venv','.pytest_cache','.ipynb_checkpoints','data','datasets','features','frame_targets','cache','runs','external'}
    with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(project.rglob('*')):
            if not path.is_file() or path.resolve()==a.output.resolve(): continue
            rel=path.relative_to(project)
            if set(rel.parts)&excluded or path.name.startswith('.env') or path.suffix in {'.pt','.pth','.ckpt','.parquet','.zip','.fs2zip','.pyc'} or path.name=='g_02500000': continue
            z.write(path,'project/'+rel.as_posix())
    with zipfile.ZipFile(a.output) as z:
        bad=z.testzip()
        if bad: raise ValueError(f'Corrupt ZIP member {bad}')
        print('Packaged',len(z.namelist()),'files:',a.output)


if __name__=='__main__': main()
