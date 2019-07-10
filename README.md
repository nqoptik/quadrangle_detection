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
Copy test data to build folder:
```
cp -r ../../computer_vision_basics_data/quadrangle_detection/build/* .
```

Run card detection:
```
./quadrangle_detection <video_file>
```
