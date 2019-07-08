# Rectangle detection
A simple rectangle detection implementation.

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
cp -r ../../../computer_vision_basics_data/card_detection/build/* .
```

Run card detection:
```
./card_detection <video_file>
```
