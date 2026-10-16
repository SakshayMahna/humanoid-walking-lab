#include "g1_policy_runner/policy_runner.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Vector3.h"

namespace g1_policy_runner
{

using std::placeholders::_1;

namespace
{
// "a,b,c" -> {"a", "b", "c"}
std::vector<std::string> split_csv(const std::string & text)
{
  std::vector<std::string> items;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    items.push_back(item);
  }
  return items;
}

// "0.1,0.2" -> {0.1f, 0.2f}
std::vector<float> parse_floats(const std::string & text)
{
  std::vector<float> values;
  for (const auto & item : split_csv(text)) {
    values.push_back(std::stof(item));
  }
  return values;
}
}  // namespace

PolicyRunner::PolicyRunner()
: Node("policy_runner")
{
  // Parameters (config/<policy>.yaml + the launch file)
  const auto policy_file = declare_parameter<std::string>("policy_file");
  clock_ = declare_parameter<std::string>("clock", "wall");
  policy_period_ = declare_parameter<double>("policy_period", 0.02);
  gait_period_ = declare_parameter<double>("gait_period", 0.6);

  // ONNX session, one thread (as Diego's node)
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_ = std::make_unique<Ort::Session>(env_, policy_file.c_str(), options);

  input_name_ = session_->GetInputNameAllocated(0, allocator_).get();
  output_name_ = session_->GetOutputNameAllocated(0, allocator_).get();
  obs_dim_ = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().back();
  action_dim_ = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().back();

  load_metadata();

  joint_pos_rel_.assign(joint_names_.size(), 0.f);
  joint_vel_.assign(joint_names_.size(), 0.f);
  last_action_.assign(action_dim_, 0.f);

  // One position-command topic per joint, in policy order (bridged to Gazebo's JointPositionController)
  for (const auto & name : joint_names_) {
    joint_publishers_.push_back(create_publisher<std_msgs::msg::Float64>("/g1/cmd_pos/" + name, 10));
  }

  twist_sub_ = create_subscription<geometry_msgs::msg::Twist>(
    "/cmd_vel", 10, std::bind(&PolicyRunner::twist_callback, this, _1));
  joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
    "/g1/joint_states", 10, std::bind(&PolicyRunner::joint_state_callback, this, _1));
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
    "/g1/imu/pelvis", 10, std::bind(&PolicyRunner::imu_callback, this, _1));
  odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    "/g1/pelvis/odometry", 10, std::bind(&PolicyRunner::odometry_callback, this, _1));

  auto period = std::chrono::duration_cast<std::chrono::nanoseconds> (
    std::chrono::duration<double>(policy_period_));

  if (clock_ == "wall")
    timer_ = create_wall_timer(period, std::bind(&PolicyRunner::policy_step, this));   // real time
  else if (clock_ == "sim")
    timer_ = create_timer(period, std::bind(&PolicyRunner::policy_step, this));        // node clock = /clock
  else
    throw std::invalid_argument("clock must be 'wall' or 'sim', got '" + clock_ + "'");

  RCLCPP_INFO(
    get_logger(), "Loaded %s: %zu observations, %zu actions, %zu joints, clock=%s, period=%.3f s",
    policy_file.c_str(), obs_dim_, action_dim_, joint_names_.size(), clock_.c_str(), policy_period_);
}

std::string PolicyRunner::metadata_string(const std::string & key)
{
  auto metadata = session_->GetModelMetadata();
  auto value = metadata.LookupCustomMetadataMapAllocated(key.c_str(), allocator_);
  if (!value) {
    throw std::runtime_error("ONNX metadata has no key '" + key + "'");
  }
  return value.get();
}

void PolicyRunner::load_metadata()
{
  joint_names_ = split_csv(metadata_string("joint_names"));
  default_pos_ = parse_floats(metadata_string("default_joint_pos"));
  action_scale_ = parse_floats(metadata_string("action_scale"));
  observation_names_ = split_csv(metadata_string("observation_names"));

  for(size_t i = 0; i < joint_names_.size(); ++i)
    index_of_[joint_names_[i]] = i;

  const size_t n = joint_names_.size();
  if (default_pos_.size() != n || action_scale_.size() != n) {
    throw std::runtime_error(
      "metadata mismatch: " + std::to_string(n) + " joint names, " +
      std::to_string(default_pos_.size()) + " default positions, " +
      std::to_string(action_scale_.size()) + " action scales");
  }
  if (action_dim_ != n) {   // e.g. walk_free_elbows.onnx: 27 actions for 29 joints
    throw std::runtime_error(
      "network outputs " + std::to_string(action_dim_) + " actions but metadata lists " +
      std::to_string(n) + " joints; this node expects one action per joint");
  }
}

void PolicyRunner::twist_callback(const geometry_msgs::msg::Twist::ConstSharedPtr msg)
{
  command_ = {
    static_cast<float>(msg->linear.x),
    static_cast<float>(msg->linear.y),
    static_cast<float>(msg->angular.z)};
}

void PolicyRunner::joint_state_callback(const sensor_msgs::msg::JointState::ConstSharedPtr msg)
{
  for (size_t k = 0; k < msg->name.size(); ++k) {   // k: Gazebo's order, i: policy order
    auto it = index_of_.find(msg->name[k]);
    if (it == index_of_.end()) continue;
    size_t i = it->second;
    joint_pos_rel_[i] = static_cast<float>(msg->position[k] - default_pos_[i]);
    joint_vel_[i] = static_cast<float>(msg->velocity[k]);
  }
}

void PolicyRunner::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
  ang_vel_[0] = msg->angular_velocity.x;
  ang_vel_[1] = msg->angular_velocity.y;
  ang_vel_[2] = msg->angular_velocity.z;

  // ROS quaternion order is x, y, z, w (MuJoCo's is w, x, y, z)
  tf2::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
  tf2::Matrix3x3 R(q);                                       // pelvis -> world rotation
  tf2::Vector3 g = R.transpose() * tf2::Vector3(0, 0, -1);   // world "down" in the pelvis frame

  gravity_[0] = g.x();
  gravity_[1] = g.y();
  gravity_[2] = g.z();
}

void PolicyRunner::odometry_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  lin_vel_[0] = msg->twist.twist.linear.x;
  lin_vel_[1] = msg->twist.twist.linear.y;
  lin_vel_[2] = msg->twist.twist.linear.z;
}

std::array<float, 2> PolicyRunner::gait_phase() const
{
  // Unitree's gait clock (Python: gait_phase() in exercises/unitree_utils.py).
  // Training: t = episode steps x 0.02 s. Here t = steps_ x policy_period_: identical at the default 0.02 s,
  // and when the rate is changed a stride still lasts gait_period_ seconds on the timer's clock.
  // Alternatives to compare later: steps_ x 0.02 (stride = 30 policy steps at any rate), or now().seconds().
  if (std::hypot(command_[0], command_[1], command_[2]) < 0.1f) {
    return {0.f, 0.f};   // stand-still command: clock off
  }
  const double t = static_cast<double>(steps_) * policy_period_;
  const double phi = std::fmod(t, gait_period_) / gait_period_;
  return {static_cast<float>(std::sin(2.0 * M_PI * phi)), static_cast<float>(std::cos(2.0 * M_PI * phi))};
}

std::vector<float> PolicyRunner::build_observation() const
{
  std::vector<float> obs;
  for (const auto& term : observation_names_) {
    if (term == "base_lin_vel")
      obs.insert(obs.end(), lin_vel_.begin(), lin_vel_.end());
    else if (term == "base_ang_vel")
      obs.insert(obs.end(), ang_vel_.begin(), ang_vel_.end());
    else if (term == "projected_gravity")
      obs.insert(obs.end(), gravity_.begin(), gravity_.end());
    else if (term == "command")
      obs.insert(obs.end(), command_.begin(), command_.end());
    else if (term == "phase") {
      auto phase = gait_phase();
      obs.insert(obs.end(), phase.begin(), phase.end());
    }
    else if (term == "joint_pos")
      obs.insert(obs.end(), joint_pos_rel_.begin(), joint_pos_rel_.end());
    else if (term == "joint_vel")
      obs.insert(obs.end(), joint_vel_.begin(), joint_vel_.end());
    else if (term == "actions")
      obs.insert(obs.end(), last_action_.begin(), last_action_.end());
    else
      throw std::runtime_error("unsupported observation term: " + term);
  }
  return obs;
}

std::vector<float> PolicyRunner::run_network(std::vector<float> & obs)
{
  if (obs.size() != obs_dim_) {
    throw std::runtime_error(
      "observation has " + std::to_string(obs.size()) + " numbers, network expects " +
      std::to_string(obs_dim_));
  }
  const std::array<int64_t, 2> shape{1, static_cast<int64_t>(obs.size())};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input = Ort::Value::CreateTensor<float>(
    memory, obs.data(), obs.size(), shape.data(), shape.size());

  const char * input_names[] = {input_name_.c_str()};
  const char * output_names[] = {output_name_.c_str()};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);

  const float * action = outputs[0].GetTensorData<float>();
  return std::vector<float>(action, action + action_dim_);
}

void PolicyRunner::policy_step()
{
  // Python equivalent: make_policy_controller() / action_to_ctrl() in exercises/policy_utils.py
  auto obs = build_observation();
  auto action = run_network(obs);

  for (size_t i = 0; i < joint_names_.size(); ++i) {
    std_msgs::msg::Float64 target;
    target.data = default_pos_[i] + action_scale_[i] * action[i];   // no clipping, as in training
    joint_publishers_[i]->publish(target);
  }

  last_action_ = action;   // raw network output goes into the next observation
  steps_++;
}

}  // namespace g1_policy_runner
