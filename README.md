# Quangrangle detection
A simple quadrangle detection implementation.

## Build project
Build project with cmake:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Run project
Clone and copy test data to build folder:
```
cd ../../
git clone https://github.com/nqoptik/computer_vision_data.git
cd quadrangle_detection/build/
cp -r ../../computer_vision_data/quadrangle_detection/build/* .
```

Run card detection:
```
./quadrangle_detection <video_file>
```
