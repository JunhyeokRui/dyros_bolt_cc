#include "dyros_bolt_lib/robot_data.h"
#include "dyros_bolt_lib/state_manager.h"
#include "wholebody_functions.h"
#include <random>
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/Joy.h>

class CustomController
{
public:
    CustomController(RobotData &rd, StateManager &stm, DataContainer &dc);
    Eigen::VectorQd getControl();

    //void taskCommandToCC(TaskCommand tc_);
    
    void computeSlow();
    void computeFast();
    void computePlanner();
    void copyRobotData(RobotData &rd_l);


    
    

    //////////////////////////////////////////// Donghyeon RL /////////////////////////////////////////
    void loadNetwork();
    void processNoise();
    void processObservation();
    void feedforwardPolicy();
    void initVariable();
    Eigen::Vector3d mat2euler(Eigen::Matrix3d mat);
    Eigen::Vector3d quat_rotate_inverse(const Eigen::Quaterniond& q, const Eigen::Vector3d& v); 

    static const int num_action = 6;
    static const int num_actuator_action = 6;
    static const int num_cur_state = 33;
    static const int num_cur_internal_state = 31;
    static const int num_state_skip = 2;
    static const int num_state_hist = 5;
    static const int num_state = num_cur_internal_state*num_state_hist+num_action*(num_state_hist-1);
    static const int num_hidden_0 = 512;
    static const int num_hidden_2 = 256;
    static const int num_hidden_4 = 128;

    static const int num_commands = 3;


    Eigen::MatrixXd policy_net_w0_;
    Eigen::MatrixXd policy_net_b0_;
    Eigen::MatrixXd policy_net_w2_;
    Eigen::MatrixXd policy_net_b2_;
    Eigen::MatrixXd policy_net_w4_;
    Eigen::MatrixXd policy_net_b4_;
    Eigen::MatrixXd action_net_w_;
    Eigen::MatrixXd action_net_b_;
    Eigen::MatrixXd hidden_layer_1;
    Eigen::MatrixXd hidden_layer_2;
    Eigen::MatrixXd hidden_layer_3;
    Eigen::MatrixXd rl_action_;

    Eigen::MatrixXd value_net_w0_;
    Eigen::MatrixXd value_net_b0_;
    Eigen::MatrixXd value_net_w2_;
    Eigen::MatrixXd value_net_b2_;
    Eigen::MatrixXd value_net_w4_;
    Eigen::MatrixXd value_net_b4_;
    Eigen::MatrixXd value_net_w_;
    Eigen::MatrixXd value_net_b_;
    Eigen::MatrixXd value_hidden_layer_1;
    Eigen::MatrixXd value_hidden_layer_2;
    Eigen::MatrixXd value_hidden_layer_3;
    double value_;

    bool stop_by_value_thres_ = false;
    Eigen::Matrix<double, MODEL_DOF, 1> q_stop_;
    float stop_start_time_;
    
    Eigen::MatrixXd state_;
    Eigen::MatrixXd state_cur_;
    Eigen::MatrixXd state_buffer_;
    Eigen::MatrixXd state_mean_;
    Eigen::MatrixXd state_var_;

    std::ofstream writeFile;

    float phase_ = 0.0;

    bool is_on_robot_ = false;
    bool is_write_file_ = true;
    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_;

    Eigen::Matrix<double, MODEL_DOF, 1> q_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_pre_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_vel_noise_;

    Eigen::Matrix<double, MODEL_DOF, 1> torque_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_spline_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_rl_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_bound_;

    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kp_;
    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kv_;

    Eigen::Vector3d base_lin_acc;
    Eigen::Vector3d base_lin_vel;
    Eigen::Vector3d base_ang_vel;
    Eigen::Vector3d base_lin_pos;
    Eigen::Vector3d projected_gravity;
    Eigen::Vector3d gravity_vector;
    Eigen::Quaterniond base_link_quat;
    


    float start_time_;
    float time_inference_pre_ = 0.0;
    float time_write_pre_ = 0.0;

    double time_cur_;
    double time_pre_;
    double action_dt_accumulate_ = 0.0;

    Eigen::Vector3d euler_angle_;

    // Joystick
    ros::NodeHandle nh_;

    void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    ros::Subscriber joy_sub_;

    double target_vel_x_ = 0.0;
    double target_vel_y_ = 0.0;
    double target_vel_z_ = 0.0;

    double policy_frequency_ = 250.0; //Hz

private:
    RobotData &rd_;
    StateManager &stm_;
    DataContainer &dc_;
    RobotData rd_cc_;
    // StateManager stm_cc_;
    Eigen::VectorQd ControlVal_;
};