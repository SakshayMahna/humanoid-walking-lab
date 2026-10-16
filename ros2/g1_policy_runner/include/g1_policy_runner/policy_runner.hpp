#ifndef G1_POLICY_RUNNER__POLICY_RUNNER_HPP_
#define G1_POLICY_RUNNER__POLICY_RUNNER_HPP_

#include <onnxruntime_cxx_api.h>

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64.hpp"

namespace g1_policy_runner
{

// Runs an mjlab-exported velocity policy on the G1 in Gazebo (Diego's g1_sim topics).
// Everything policy-specific comes from the ONNX metadata, so the same node runs
// walk.onnx (99 observations) and Unitree's policy.onnx (98 observations).
class PolicyRunner : public rclcpp::Node
{
public:
  PolicyRunner();

private:
  // --- Setup ---
  void load_metadata();                                   
  std::string metadata_string(const std::string & key);   // ONNX metadata value (comma-separated text)

  // --- Sensor callbacks: store the latest values, in POLICY joint order ---
  void twist_callback(const geometry_msgs::msg::Twist::ConstSharedPtr msg);
  void joint_state_callback(const sensor_msgs::msg::JointState::ConstSharedPtr msg);   
  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg);                  
  void odometry_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg);           

  // --- Policy step (every 20 ms) ---
  std::array<float, 2> gait_phase() const;                
  std::vector<float> build_observation() const;           
  std::vector<float> run_network(std::vector<float> & obs);
  void policy_step();                                     

  // --- From the ONNX metadata (policy joint order) ---
  std::vector<std::string> joint_names_;          // "joint_names"
  std::vector<float> default_pos_;                // "default_joint_pos" (rad)
  std::vector<float> action_scale_;               // "action_scale"
  std::vector<std::string> observation_names_;    // "observation_names": the terms, in order
  std::unordered_map<std::string, size_t> index_of_;   // joint name -> index in joint_names_

  // --- Parameters ---
  std::string clock_;            // "wall" or "sim"
  double policy_period_;         // s between policy steps; training: 0.02 (50 Hz = decimation 4 x 5 ms)
  double gait_period_;           // s, Unitree's phase term only

  // --- Latest inputs (written by callbacks, read by policy_step) ---
  std::array<float, 3> command_{};            // vx, vy, wz from /cmd_vel
  std::array<float, 3> lin_vel_{};            // pelvis-frame linear velocity (odometry)
  std::array<float, 3> ang_vel_{};            // pelvis-frame angular velocity (IMU)
  std::array<float, 3> gravity_{};            // "down" in the pelvis frame
  std::vector<float> joint_pos_rel_;          // joint angle - default, policy order
  std::vector<float> joint_vel_;              // joint velocity, policy order
  std::vector<float> last_action_;            // previous raw network output
  uint64_t steps_ = 0;                        // policy steps since start

  // --- ROS interfaces ---
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  std::vector<rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr> joint_publishers_;   // policy order
  rclcpp::TimerBase::SharedPtr timer_;

  // --- ONNX Runtime ---
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "g1_policy_runner"};
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::string input_name_, output_name_;
  size_t obs_dim_ = 0, action_dim_ = 0;       // from the network's input/output shapes
};

}  // namespace g1_policy_runner

#endif  // G1_POLICY_RUNNER__POLICY_RUNNER_HPP_
