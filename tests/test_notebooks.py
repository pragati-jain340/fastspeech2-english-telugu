from pathlib import Path
import ast
import nbformat


def test_notebooks_valid_and_python_cells_compile():
    paths=sorted((Path(__file__).resolve().parents[1]/'notebooks').glob('*.ipynb'))
    assert len(paths)==5
    for p in paths:
        n=nbformat.read(p,as_version=4);nbformat.validate(n)
        assert 'REPO_URL' in n.cells[1].source
        for c in n.cells:
            if c.cell_type=='code':
                ast.parse(c.source)
                assert c.execution_count is None and c.outputs==[]
