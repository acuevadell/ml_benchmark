# Machine Learning benchmark

Required Packages:
```
sudo apt install python3
sudo apt install python3-pip
pip install tensorflow --break-system-packages
pip install matplotlib --break-system-packages

#pip install pyRAPL --break-system-packages
#pip install pyJoules --break-system-packages
#pip install energyusage --break-system-packages
```

Run benchmark:
```
python3 benchmark.py
jq . result.json
jq . predictions.json
```

## MLPerf

required
```
pip install mlcommons-loadgen --break-system-packages
pip install opencv-python --break-system-packages
#pip install cython --break-system-packages
```


```
python main.py --model resnet50 --dataset-path /home/labdev/code/dogcat --scenario Offline --mode performance
```