# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.5.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.5.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/JohnsonJohnson/Desktop/Desktop/multishots

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/JohnsonJohnson/Desktop/Desktop/multishots/build

# Include any dependencies generated for this target.
include CMakeFiles/123p.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/123p.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/123p.dir/flags.make

CMakeFiles/123p.dir/main.cpp.o: CMakeFiles/123p.dir/flags.make
CMakeFiles/123p.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/JohnsonJohnson/Desktop/Desktop/multishots/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/123p.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/123p.dir/main.cpp.o -c /Users/JohnsonJohnson/Desktop/Desktop/multishots/main.cpp

CMakeFiles/123p.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/123p.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/JohnsonJohnson/Desktop/Desktop/multishots/main.cpp > CMakeFiles/123p.dir/main.cpp.i

CMakeFiles/123p.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/123p.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/JohnsonJohnson/Desktop/Desktop/multishots/main.cpp -o CMakeFiles/123p.dir/main.cpp.s

CMakeFiles/123p.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/123p.dir/main.cpp.o.requires

CMakeFiles/123p.dir/main.cpp.o.provides: CMakeFiles/123p.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/123p.dir/build.make CMakeFiles/123p.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/123p.dir/main.cpp.o.provides

CMakeFiles/123p.dir/main.cpp.o.provides.build: CMakeFiles/123p.dir/main.cpp.o


CMakeFiles/123p.dir/descriptor.cpp.o: CMakeFiles/123p.dir/flags.make
CMakeFiles/123p.dir/descriptor.cpp.o: ../descriptor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/JohnsonJohnson/Desktop/Desktop/multishots/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/123p.dir/descriptor.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/123p.dir/descriptor.cpp.o -c /Users/JohnsonJohnson/Desktop/Desktop/multishots/descriptor.cpp

CMakeFiles/123p.dir/descriptor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/123p.dir/descriptor.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/JohnsonJohnson/Desktop/Desktop/multishots/descriptor.cpp > CMakeFiles/123p.dir/descriptor.cpp.i

CMakeFiles/123p.dir/descriptor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/123p.dir/descriptor.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/JohnsonJohnson/Desktop/Desktop/multishots/descriptor.cpp -o CMakeFiles/123p.dir/descriptor.cpp.s

CMakeFiles/123p.dir/descriptor.cpp.o.requires:

.PHONY : CMakeFiles/123p.dir/descriptor.cpp.o.requires

CMakeFiles/123p.dir/descriptor.cpp.o.provides: CMakeFiles/123p.dir/descriptor.cpp.o.requires
	$(MAKE) -f CMakeFiles/123p.dir/build.make CMakeFiles/123p.dir/descriptor.cpp.o.provides.build
.PHONY : CMakeFiles/123p.dir/descriptor.cpp.o.provides

CMakeFiles/123p.dir/descriptor.cpp.o.provides.build: CMakeFiles/123p.dir/descriptor.cpp.o


CMakeFiles/123p.dir/gauss.cpp.o: CMakeFiles/123p.dir/flags.make
CMakeFiles/123p.dir/gauss.cpp.o: ../gauss.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/JohnsonJohnson/Desktop/Desktop/multishots/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/123p.dir/gauss.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/123p.dir/gauss.cpp.o -c /Users/JohnsonJohnson/Desktop/Desktop/multishots/gauss.cpp

CMakeFiles/123p.dir/gauss.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/123p.dir/gauss.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/JohnsonJohnson/Desktop/Desktop/multishots/gauss.cpp > CMakeFiles/123p.dir/gauss.cpp.i

CMakeFiles/123p.dir/gauss.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/123p.dir/gauss.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/JohnsonJohnson/Desktop/Desktop/multishots/gauss.cpp -o CMakeFiles/123p.dir/gauss.cpp.s

CMakeFiles/123p.dir/gauss.cpp.o.requires:

.PHONY : CMakeFiles/123p.dir/gauss.cpp.o.requires

CMakeFiles/123p.dir/gauss.cpp.o.provides: CMakeFiles/123p.dir/gauss.cpp.o.requires
	$(MAKE) -f CMakeFiles/123p.dir/build.make CMakeFiles/123p.dir/gauss.cpp.o.provides.build
.PHONY : CMakeFiles/123p.dir/gauss.cpp.o.provides

CMakeFiles/123p.dir/gauss.cpp.o.provides.build: CMakeFiles/123p.dir/gauss.cpp.o


CMakeFiles/123p.dir/histdescriptor.cpp.o: CMakeFiles/123p.dir/flags.make
CMakeFiles/123p.dir/histdescriptor.cpp.o: ../histdescriptor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/JohnsonJohnson/Desktop/Desktop/multishots/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/123p.dir/histdescriptor.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/123p.dir/histdescriptor.cpp.o -c /Users/JohnsonJohnson/Desktop/Desktop/multishots/histdescriptor.cpp

CMakeFiles/123p.dir/histdescriptor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/123p.dir/histdescriptor.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/JohnsonJohnson/Desktop/Desktop/multishots/histdescriptor.cpp > CMakeFiles/123p.dir/histdescriptor.cpp.i

CMakeFiles/123p.dir/histdescriptor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/123p.dir/histdescriptor.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/JohnsonJohnson/Desktop/Desktop/multishots/histdescriptor.cpp -o CMakeFiles/123p.dir/histdescriptor.cpp.s

CMakeFiles/123p.dir/histdescriptor.cpp.o.requires:

.PHONY : CMakeFiles/123p.dir/histdescriptor.cpp.o.requires

CMakeFiles/123p.dir/histdescriptor.cpp.o.provides: CMakeFiles/123p.dir/histdescriptor.cpp.o.requires
	$(MAKE) -f CMakeFiles/123p.dir/build.make CMakeFiles/123p.dir/histdescriptor.cpp.o.provides.build
.PHONY : CMakeFiles/123p.dir/histdescriptor.cpp.o.provides

CMakeFiles/123p.dir/histdescriptor.cpp.o.provides.build: CMakeFiles/123p.dir/histdescriptor.cpp.o


CMakeFiles/123p.dir/ReidDescriptor.cpp.o: CMakeFiles/123p.dir/flags.make
CMakeFiles/123p.dir/ReidDescriptor.cpp.o: ../ReidDescriptor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/JohnsonJohnson/Desktop/Desktop/multishots/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/123p.dir/ReidDescriptor.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/123p.dir/ReidDescriptor.cpp.o -c /Users/JohnsonJohnson/Desktop/Desktop/multishots/ReidDescriptor.cpp

CMakeFiles/123p.dir/ReidDescriptor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/123p.dir/ReidDescriptor.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/JohnsonJohnson/Desktop/Desktop/multishots/ReidDescriptor.cpp > CMakeFiles/123p.dir/ReidDescriptor.cpp.i

CMakeFiles/123p.dir/ReidDescriptor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/123p.dir/ReidDescriptor.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/JohnsonJohnson/Desktop/Desktop/multishots/ReidDescriptor.cpp -o CMakeFiles/123p.dir/ReidDescriptor.cpp.s

CMakeFiles/123p.dir/ReidDescriptor.cpp.o.requires:

.PHONY : CMakeFiles/123p.dir/ReidDescriptor.cpp.o.requires

CMakeFiles/123p.dir/ReidDescriptor.cpp.o.provides: CMakeFiles/123p.dir/ReidDescriptor.cpp.o.requires
	$(MAKE) -f CMakeFiles/123p.dir/build.make CMakeFiles/123p.dir/ReidDescriptor.cpp.o.provides.build
.PHONY : CMakeFiles/123p.dir/ReidDescriptor.cpp.o.provides

CMakeFiles/123p.dir/ReidDescriptor.cpp.o.provides.build: CMakeFiles/123p.dir/ReidDescriptor.cpp.o


CMakeFiles/123p.dir/watershed.cpp.o: CMakeFiles/123p.dir/flags.make
CMakeFiles/123p.dir/watershed.cpp.o: ../watershed.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/JohnsonJohnson/Desktop/Desktop/multishots/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/123p.dir/watershed.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/123p.dir/watershed.cpp.o -c /Users/JohnsonJohnson/Desktop/Desktop/multishots/watershed.cpp

CMakeFiles/123p.dir/watershed.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/123p.dir/watershed.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/JohnsonJohnson/Desktop/Desktop/multishots/watershed.cpp > CMakeFiles/123p.dir/watershed.cpp.i

CMakeFiles/123p.dir/watershed.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/123p.dir/watershed.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/JohnsonJohnson/Desktop/Desktop/multishots/watershed.cpp -o CMakeFiles/123p.dir/watershed.cpp.s

CMakeFiles/123p.dir/watershed.cpp.o.requires:

.PHONY : CMakeFiles/123p.dir/watershed.cpp.o.requires

CMakeFiles/123p.dir/watershed.cpp.o.provides: CMakeFiles/123p.dir/watershed.cpp.o.requires
	$(MAKE) -f CMakeFiles/123p.dir/build.make CMakeFiles/123p.dir/watershed.cpp.o.provides.build
.PHONY : CMakeFiles/123p.dir/watershed.cpp.o.provides

CMakeFiles/123p.dir/watershed.cpp.o.provides.build: CMakeFiles/123p.dir/watershed.cpp.o


# Object files for target 123p
123p_OBJECTS = \
"CMakeFiles/123p.dir/main.cpp.o" \
"CMakeFiles/123p.dir/descriptor.cpp.o" \
"CMakeFiles/123p.dir/gauss.cpp.o" \
"CMakeFiles/123p.dir/histdescriptor.cpp.o" \
"CMakeFiles/123p.dir/ReidDescriptor.cpp.o" \
"CMakeFiles/123p.dir/watershed.cpp.o"

# External object files for target 123p
123p_EXTERNAL_OBJECTS =

123p: CMakeFiles/123p.dir/main.cpp.o
123p: CMakeFiles/123p.dir/descriptor.cpp.o
123p: CMakeFiles/123p.dir/gauss.cpp.o
123p: CMakeFiles/123p.dir/histdescriptor.cpp.o
123p: CMakeFiles/123p.dir/ReidDescriptor.cpp.o
123p: CMakeFiles/123p.dir/watershed.cpp.o
123p: CMakeFiles/123p.dir/build.make
123p: /usr/local/lib/libopencv_videostab.2.4.13.dylib
123p: /usr/local/lib/libopencv_ts.a
123p: /usr/local/lib/libopencv_superres.2.4.13.dylib
123p: /usr/local/lib/libopencv_stitching.2.4.13.dylib
123p: /usr/local/lib/libopencv_contrib.2.4.13.dylib
123p: /usr/local/lib/libopencv_nonfree.2.4.13.dylib
123p: /usr/local/lib/libopencv_ocl.2.4.13.dylib
123p: /usr/local/lib/libopencv_gpu.2.4.13.dylib
123p: /usr/local/lib/libopencv_photo.2.4.13.dylib
123p: /usr/local/lib/libopencv_objdetect.2.4.13.dylib
123p: /usr/local/lib/libopencv_legacy.2.4.13.dylib
123p: /usr/local/lib/libopencv_video.2.4.13.dylib
123p: /usr/local/lib/libopencv_ml.2.4.13.dylib
123p: /usr/local/lib/libopencv_calib3d.2.4.13.dylib
123p: /usr/local/lib/libopencv_features2d.2.4.13.dylib
123p: /usr/local/lib/libopencv_highgui.2.4.13.dylib
123p: /usr/local/lib/libopencv_imgproc.2.4.13.dylib
123p: /usr/local/lib/libopencv_flann.2.4.13.dylib
123p: /usr/local/lib/libopencv_core.2.4.13.dylib
123p: CMakeFiles/123p.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/JohnsonJohnson/Desktop/Desktop/multishots/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable 123p"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/123p.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/123p.dir/build: 123p

.PHONY : CMakeFiles/123p.dir/build

CMakeFiles/123p.dir/requires: CMakeFiles/123p.dir/main.cpp.o.requires
CMakeFiles/123p.dir/requires: CMakeFiles/123p.dir/descriptor.cpp.o.requires
CMakeFiles/123p.dir/requires: CMakeFiles/123p.dir/gauss.cpp.o.requires
CMakeFiles/123p.dir/requires: CMakeFiles/123p.dir/histdescriptor.cpp.o.requires
CMakeFiles/123p.dir/requires: CMakeFiles/123p.dir/ReidDescriptor.cpp.o.requires
CMakeFiles/123p.dir/requires: CMakeFiles/123p.dir/watershed.cpp.o.requires

.PHONY : CMakeFiles/123p.dir/requires

CMakeFiles/123p.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/123p.dir/cmake_clean.cmake
.PHONY : CMakeFiles/123p.dir/clean

CMakeFiles/123p.dir/depend:
	cd /Users/JohnsonJohnson/Desktop/Desktop/multishots/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/JohnsonJohnson/Desktop/Desktop/multishots /Users/JohnsonJohnson/Desktop/Desktop/multishots /Users/JohnsonJohnson/Desktop/Desktop/multishots/build /Users/JohnsonJohnson/Desktop/Desktop/multishots/build /Users/JohnsonJohnson/Desktop/Desktop/multishots/build/CMakeFiles/123p.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/123p.dir/depend

