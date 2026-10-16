#include "g1_policy_runner/policy_runner.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<g1_policy_runner::PolicyRunner>());
  rclcpp::shutdown();
  return 0;
}
