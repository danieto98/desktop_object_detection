# desktop_object_detection

Object detection and situation awareness ROS package using Kinect data for 3 common desktop objects

## System Architecture

This is a UML diagram of the overall system architecture:

<p align="center">
<img src="https://github.com/danieto98/desktop_object_detection/blob/master/desktop_object_detection(UML).png">
</p>

## Installation

### Requirements

#### Robot Operating System (ROS)

You must have a working ROS installation. Make sure you install the full desktop version for either ROS [Kinectic](http://wiki.ros.org/kinetic/Installation) or [Melodic](http://wiki.ros.org/melodic/Installation).

#### rtabmap_ros

This is a ROS wrapper for the rtabmap library used to synchronize the Kinect messages. Install it using the following command, replacing `<distro>` with your ROS installation (kinetic or melodic):

```
sudo apt-get install ros-<distro>rtabmap ros-<distro>-rtabmap-ros
```

#### libfreenect

This library provides drivers for the Microsoft Kinect.

You will first need to clone the libfreenect library from source. Open up a terminal in your desired directory and run:

```
git clone https://github.com/OpenKinect/libfreenect
cd libfreenect
```

Before installing the source, modify the CMakeLists.txt file in the repository's root directory by adding the following line:

```
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11)
```

Now make and install the library:

```
mkdir build
cd build
cmake -L ..
make
sudo make install
```

#### Python libraries

Install using pip2 (for Python 2.7):

```
pip2 install --upgrade tensorflow
pip2 install numpy
pip2 install matplotlib
pip2 install openpyxl
pip2 install scikit-image
```

### desktop_object_detection

First, create a catkin workspace on your desired location:

```
source /opt/ros/melodic/setup.bash
mkdir -p catkin_ws/src
cd catkin_ws/
catkin_make
source devel/setup.bash
```

Now, clone the freenect_stack repository:

```
cd src
git clone https://github.com/ros-drivers/freenect_stack
```

Clone this repository there too:

```
git clone https://github.com/danieto98/desktop_object_detection
```

Make the catkin workspace and source it:

```
cd ..
catkin_make
source devel/setup.bash
```

Make all the ROS Python files in the repository executable:

```
find src/desktop_object_detection/src -type f -exec chmod +x {} \;
chmod +x src/desktop_object_detection/test/result_message_stream.py
```

## Usage

### Record Kinect Data

Connect the Kinect sensor by USB to the PC running ROS. Use the following command to test whether the device is recognized:

```
lsusb | grep Xbox
```

You should see three devices there (Xbox NUI Camera, Xbox NUI Motor, Xbox NUI Audio). Once you have seen them, open a terminal in the root of your catkin workspace and run:

```
source devel/setup.bash
```

Run the following command by replacing `<prefix>` with your desired prefix for the bag file that will be saved as `absolute_path/name` (e.g. `/home/user/bag1` where `bag1` is not a folder):

```
roslaunch desktop_object_detection record_bag.launch bag_prefix:=<prefix>
```

Kill at any desired moment all the processes to save the bag file by inputting `Ctrl+C` in the terminal that is running them.

### Play Bag and Run

Download and extract the files for the trained model of the Convolutional Neural Network (CNN) from [here](https://drive.google.com/open?id=1Ruqc53FRV53kMj4XMkbf9ik6u8gZjcc7).

To run all the nodes while playing back the bag file saved using the previous procedure, use the following command by replacing `<log_path>` with the absolute path of the Excel log file that will be created (use a .xlsx extension), `<bag_filename>` with the absolute path to the recorded bag file, and `<model_dir>` with the absolute path to the directory of the CNN:

```
roslaunch desktop_object_detection run_bag.launch log_filepath:=<log_path> filename:=<bag_filename> model_path:=<model_dir>
```