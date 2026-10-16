# ROS 2 Jazzy + Gazebo Harmonic desktop (VNC) for running Diego's g1_sim / g1_locomotion.
# Versions match 3p/2025-tfg-diego-lopez/README.md: Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic, ONNX Runtime 1.17.1.
# The base image already ships ros-jazzy-ros-gz 1.0.24 with Gazebo Sim 8 (Harmonic).
FROM tiryoh/ros2-desktop-vnc:jazzy

# ONNX Runtime C++. g1_locomotion/CMakeLists.txt links /usr/local/lib/libonnxruntime.so directly.
# Diego's README uses the x64 build; pick the build that matches this machine (aarch64 on Apple Silicon).
ARG ORT_VERSION=1.17.1
RUN case "$(uname -m)" in \
        aarch64) ORT_ARCH=aarch64 ;; \
        x86_64)  ORT_ARCH=x64 ;; \
        *) echo "unsupported architecture: $(uname -m)" && exit 1 ;; \
    esac \
    && ORT_NAME=onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION} \
    && curl -fsSL -o /tmp/ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_NAME}.tgz \
    && tar -xzf /tmp/ort.tgz -C /tmp \
    && cp -r /tmp/${ORT_NAME}/include/* /usr/local/include/ \
    && cp /tmp/${ORT_NAME}/lib/libonnxruntime.so* /usr/local/lib/ \
    && ldconfig \
    && rm -rf /tmp/ort.tgz /tmp/${ORT_NAME}
