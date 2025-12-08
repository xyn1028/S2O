# S2O
SFPG-Diff: SAR-to-Optical Translation via Spatial-Frequency Integrated Partial Guidance
create a new environment:
$ conda env create -f environment.yml

install [softpool](https://github.com/alexandrosstergiou/SoftPool):
$ cd SoftPool/pytorch
$ make install

Training:
# stage 1 
python main.py --config 'config/SEN12_256_s1.json'

# stage 2 
python main.py --config 'config/SEN12_256_s2_1step.json'

Test:
python main.py --config 'config/SEN12_256_s2_test.json' --phase 'val'  --seed 1
