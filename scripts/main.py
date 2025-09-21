from pathlib import Path
from utils import main

if __name__ == '__main__':
    Path('output').mkdir(exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    main()
