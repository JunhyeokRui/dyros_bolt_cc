#include "cc.h"
#include <X11/Xlib.h>
#include <rbdl/Kinematics.h>
#include <yaml-cpp/yaml.h>

//ANCHOR - for policy test
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


using namespace DYROS_BOLT;

CustomController::CustomController(RobotData &rd, StateManager &stm, DataContainer &dc) : rd_(rd), stm_(stm), dc_(dc)//, wbc_(dc.wbc_)
{
    ControlVal_.setZero();

    if (is_write_file_)
    {
        if (is_on_robot_)
        {
            writeFile.open("/home/dyros/dyros_bolt_ws/src/dyros_bolt_cc/result/data.csv", std::ofstream::out | std::ofstream::app);
        }
        else
        {
            writeFile.open("/home/dyros/dyros_bolt_ws/src/dyros_bolt_cc/result/data.csv", std::ofstream::out | std::ofstream::app);
        }
        writeFile << std::fixed << std::setprecision(8);
    }
    initVariable();
    loadNetwork();

    joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 10, &CustomController::joyCallback, this);
}

Eigen::VectorQd CustomController::getControl()
{
    return ControlVal_;
}

Eigen::Vector3d CustomController::mat2euler(Eigen::Matrix3d mat)
{
    Eigen::Vector3d euler;

    double cy = std::sqrt(mat(2, 2) * mat(2, 2) + mat(1, 2) * mat(1, 2));
    if (cy > std::numeric_limits<double>::epsilon())
    {
        euler(2) = -atan2(mat(0, 1), mat(0, 0));
        euler(1) =  -atan2(-mat(0, 2), cy);
        euler(0) = -atan2(mat(1, 2), mat(2, 2));
    }
    else
    {
        euler(2) = -atan2(-mat(1, 0), mat(1, 1));
        euler(1) =  -atan2(-mat(0, 2), cy);
        euler(0) = 0.0;
    }
    return euler;
}

Eigen::Vector3d CustomController::quat_rotate_inverse(const Eigen::Quaterniond& q, const Eigen::Vector3d& v) 
{
    Eigen::Vector3d q_vec = q.vec();
    double q_w = q.w();

    Eigen::Vector3d a = v * (2.0 * q_w * q_w - 1.0);
    Eigen::Vector3d b = q_vec.cross(v) * q_w * 2.0;
    Eigen::Vector3d c = q_vec * (q_vec.dot(v) * 2.0);

    return a - b + c;
}

Eigen::MatrixXd CustomController::applyELU(const Eigen::MatrixXd& input, double alpha = 1.0) {
    Eigen::MatrixXd output = input;

    for (int i = 0; i < output.rows(); ++i) {
        for (int j = 0; j < output.cols(); ++j) {
            double x = output(i, j);
            output(i, j) = (x > 0) ? x : alpha * (exp(x) - 1);
        }
    }

    return output;
}

void CustomController::loadNetwork() //rui weight 불러오기 weight TocabiRL 파일 저장된 12개 파일 저장해주면 됨
{
    state_.setZero();
    rl_action_.setZero();


    string cur_path = "/home/dyros/bolt_ws/src/dyros_bolt_cc/";

    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/";
    }
    std::ifstream file[18];
    file[0].open(cur_path+"weight/actor_0_weight.txt", std::ios::in);
    file[1].open(cur_path+"weight/actor_0_bias.txt", std::ios::in);
    file[2].open(cur_path+"weight/actor_2_weight.txt", std::ios::in);
    file[3].open(cur_path+"weight/actor_2_bias.txt", std::ios::in);
    file[4].open(cur_path+"weight/actor_4_weight.txt", std::ios::in);
    file[5].open(cur_path+"weight/actor_4_bias.txt", std::ios::in);
    file[6].open(cur_path+"weight/actor_6_weight.txt", std::ios::in);
    file[7].open(cur_path+"weight/actor_6_bias.txt", std::ios::in);
    file[8].open(cur_path+"weight/critic_0_weight.txt", std::ios::in);
    file[9].open(cur_path+"weight/critic_0_bias.txt", std::ios::in);
    file[10].open(cur_path+"weight/critic_2_weight.txt", std::ios::in);
    file[11].open(cur_path+"weight/critic_2_bias.txt", std::ios::in);
    file[12].open(cur_path+"weight/critic_4_weight.txt", std::ios::in);
    file[13].open(cur_path+"weight/critic_4_bias.txt", std::ios::in);
    file[14].open(cur_path+"weight/critic_6_weight.txt", std::ios::in);
    file[15].open(cur_path+"weight/critic_6_bias.txt", std::ios::in);
    file[16].open(cur_path+"weight/obs_mean_fixed.txt", std::ios::in);
    file[17].open(cur_path+"weight/obs_variance_fixed.txt", std::ios::in);

    if(!file[0].is_open())
    {
        std::cout << "CAN NOT FIND THE WEIGHT FILE" << std::endl;
    }

    float temp;
    int row = 0;
    int col = 0;
    while(!file[0].eof() && row != policy_net_w0_.rows())
    {
        file[0] >> temp;
        if(temp != '\n')
        {
            policy_net_w0_(row, col) = temp;
            col ++;
            if (col == policy_net_w0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[1].eof() && row != policy_net_b0_.rows())
    {
        file[1] >> temp;
        if(temp != '\n')
        {
            policy_net_b0_(row, col) = temp;
            col ++;
            if (col == policy_net_b0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[2].eof() && row != policy_net_w2_.rows())
    {
        file[2] >> temp;
        if(temp != '\n')
        {
            policy_net_w2_(row, col) = temp;
            col ++;
            if (col == policy_net_w2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[3].eof() && row != policy_net_b2_.rows())
    {
        file[3] >> temp;
        if(temp != '\n')
        {
            policy_net_b2_(row, col) = temp;
            col ++;
            if (col == policy_net_b2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[4].eof() && row != policy_net_w4_.rows())
    {
        file[4] >> temp;
        if(temp != '\n')
        {
            policy_net_w4_(row, col) = temp;
            col ++;
            if (col == policy_net_w4_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[5].eof() && row != policy_net_b4_.rows())
    {
        file[5] >> temp;
        if(temp != '\n')
        {
            policy_net_b4_(row, col) = temp;
            col ++;
            if (col == policy_net_b4_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[6].eof() && row != action_net_w_.rows())
    {
        file[6] >> temp;
        if(temp != '\n')
        {
            action_net_w_(row, col) = temp;
            col ++;
            if (col == action_net_w_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[7].eof() && row != action_net_b_.rows())
    {
        file[7] >> temp;
        if(temp != '\n')
        {
            action_net_b_(row, col) = temp;
            col ++;
            if (col == action_net_b_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[8].eof() && row != value_net_w0_.rows())
    {
        file[8] >> temp;
        if(temp != '\n')
        {
            value_net_w0_(row, col) = temp;
            col ++;
            if (col == value_net_w0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[9].eof() && row != value_net_b0_.rows())
    {
        file[9] >> temp;
        if(temp != '\n')
        {
            value_net_b0_(row, col) = temp;
            col ++;
            if (col == value_net_b0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[10].eof() && row != value_net_w2_.rows())
    {
        file[10] >> temp;
        if(temp != '\n')
        {
            value_net_w2_(row, col) = temp;
            col ++;
            if (col == value_net_w2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[11].eof() && row != value_net_b2_.rows())
    {
        file[11] >> temp;
        if(temp != '\n')
        {
            value_net_b2_(row, col) = temp;
            col ++;
            if (col == value_net_b2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[12].eof() && row != value_net_w4_.rows())
    {
        file[12] >> temp;
        if(temp != '\n')
        {
            value_net_w4_(row, col) = temp;
            col ++;
            if (col == value_net_w4_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[13].eof() && row != value_net_b4_.rows())
    {
        file[13] >> temp;
        if(temp != '\n')
        {
            value_net_b4_(row, col) = temp;
            col ++;
            if (col == value_net_b4_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[14].eof() && row != value_net_w_.rows())
    {
        file[14] >> temp;
        if(temp != '\n')
        {
            value_net_w_(row, col) = temp;
            col ++;
            if (col == value_net_w_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[15].eof() && row != value_net_b_.rows())
    {
        file[15] >> temp;
        if(temp != '\n')
        {
            value_net_b_(row, col) = temp;
            col ++;
            if (col == value_net_b_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[16].eof() && row != state_mean_.rows())
    {
        file[16] >> temp;
        if(temp != '\n')
        {
            state_mean_(row, col) = temp;
            col ++;
            if (col == state_mean_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[17].eof() && row != state_var_.rows())
    {
        file[17] >> temp;
        if(temp != '\n')
        {
            state_var_(row, col) = temp;
            col ++;
            if (col == state_var_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    
}

void CustomController::initVariable() //rui 변수 초기화
{    
    policy_net_w0_.resize(num_hidden_0, num_state);
    policy_net_b0_.resize(num_hidden_0, 1);
    policy_net_w2_.resize(num_hidden_2, num_hidden_0);
    policy_net_b2_.resize(num_hidden_2, 1);
    policy_net_w4_.resize(num_hidden_4, num_hidden_2);
    policy_net_b4_.resize(num_hidden_4, 1);
    action_net_w_.resize(num_action, num_hidden_4);
    action_net_b_.resize(num_action, 1);

    hidden_layer_1.resize(num_hidden_0, 1);
    hidden_layer_2.resize(num_hidden_2, 1);
    hidden_layer_3.resize(num_hidden_4, 1);
    rl_action_.resize(num_action, 1);

    value_net_w0_.resize(num_hidden_0, num_state);
    value_net_b0_.resize(num_hidden_0, 1);
    value_net_w2_.resize(num_hidden_2, num_hidden_0);
    value_net_b2_.resize(num_hidden_2, 1);
    value_net_w4_.resize(num_hidden_4, num_hidden_2);
    value_net_b4_.resize(num_hidden_4, 1);
    value_net_w_.resize(1, num_hidden_4);
    value_net_b_.resize(1, 1);
    
    value_hidden_layer_1.resize(num_hidden_0, 1);
    value_hidden_layer_2.resize(num_hidden_2, 1);
    value_hidden_layer_3.resize(num_hidden_4, 1);
    
    state_cur_.resize(num_cur_state, 1);
    state_cur_clipped_.resize(num_cur_state, 1);
    
    state_.resize(num_state, 1);

    state_buffer_.resize(num_cur_state*num_state_skip*num_state_hist, 1);
    state_mean_.resize(num_cur_state, 1);
    state_var_.resize(num_cur_state, 1);

    q_dot_lpf_.setZero();

    torque_bound_ << 25, 25, 25, 
                     25, 25, 25;  
    // torque_bound_ << 2.5, 2.5, 2.5, 
    //                  2.5, 2.5, 2.5; 
    obs_bound_ << 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 
                 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100;
    
    q_init_ << 0.0000,  0.2000, -0.4000,  0.0000,  0.2000, -0.4000;
    // q_init_ << 0.2, -1.8, -0.4,
    //            0.2, -1.8, -0.4;

    kp_.setZero();
    kv_.setZero();

    kp_.diagonal() <<   0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0;
    kv_.diagonal() <<   0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0;
}

void CustomController::processNoise() //rui noise 만들어주기
{
    time_cur_ = rd_cc_.control_time_us_ / 1e6;
    // if (is_on_robot_)
    // {
    //     q_vel_noise_ = rd_cc_.q_dot_virtual_.segment(6,MODEL_DOF);
    //     q_noise_= rd_cc_.q_virtual_.segment(6,MODEL_DOF);
    //     if (time_cur_ - time_pre_ > 0.0)
    //     {
    //         q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
    //     }
    //     else
    //     {
    //         q_dot_lpf_ = q_dot_lpf_;
    //     }

        
    // }
    // else
    // {
    //     std::random_device rd;  
    //     std::mt19937 gen(rd());
    //     std::uniform_real_distribution<> dis(-0.00001, 0.00001);
    //     for (int i = 0; i < MODEL_DOF; i++) {
    //         // q_noise_(i) = rd_cc_.q_virtual_(6+i) + dis(gen);
    //         q_noise_(i) = rd_cc_.q_virtual_(6+i);
    //     }
    //     if (time_cur_ - time_pre_ > 0.0)
    //     {
    //         q_vel_noise_ = (q_noise_ - q_noise_pre_) / (time_cur_ - time_pre_);
    //         q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
    //     }
    //     else
    //     {
    //         q_vel_noise_ = q_vel_noise_;
    //         q_dot_lpf_ = q_dot_lpf_;
    //     }
    //     q_noise_pre_ = q_noise_;

        

    // }
    // q_vel_noise_ = rd_cc_.q_dot_virtual_.segment(6,MODEL_DOF);
    // q_noise_= rd_cc_.q_virtual_.segment(6,MODEL_DOF);

//rui - for debug start
    // std::cout << "rd_cc_.q_dot_virtual_" << std::endl;
    // std::cout << rd_cc_.q_dot_virtual_ << std::endl;
    // std::cout << "rd_cc_.q_virtual_" << std::endl;
    // std::cout << rd_cc_.q_virtual_ << std::endl;
    // std::cout << "q_noise_" << std::endl;
    // std::cout << q_noise_ << std::endl;
    // std::cout << "q_vel_noise_" << std::endl;
    // std::cout << q_vel_noise_ << std::endl;

//rui - for debug end

}

void CustomController::processObservation() //rui observation 만들어주기 
{
    /*
    obs_buf
    size --> 2 self.contacts,
    size --> 1 self.base_z,
    size --> 3 self.base_lin_vel * self.obs_scales.lin_vel,
    size --> 3 self.base_ang_vel * self.obs_scales.ang_vel,
    size --> 3 self.projected_gravity,
    size --> 3 self.commands[:, :3] * self.commands_scale,
    size --> 6 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    size --> 6 self.dof_vel * self.obs_scales.dof_vel,
    size --> 6 self.actions
    
    total 33
    */
    
    int data_idx = 0;
    double obs_scales_lin_vel = 1.0;
    double obs_scales_ang_vel = 0.25;
    double commands_scale_target_vel_x_ = 1.0;
    double commands_scale_target_vel_y_ = 1.0;
    double commands_scale_ang_vel_yaw = 0.25;
    double obs_scales_dof_pos = 1.0;
    double obs_scales_dof_vel = 0.05;
    // base_lin_vel = rd_cc_.imu_lin_vel; 
    // base_ang_vel = rd_cc_.imu_ang_vel;
    base_link_quat = rd_cc_.base_link_xquat_rd;
    gravity_vector << 0, 0, -1;
    // std::cout << "base_link_quat" << std::endl;
    // std::cout << base_link_quat.x() << " " << base_link_quat.y() << " " << base_link_quat.z() << " " << base_link_quat.w() << std::endl;
    projected_gravity = quat_rotate_inverse(base_link_quat, gravity_vector);
    q_dot_cc = stm_.q_vel_cc_.segment(0,MODEL_DOF);
    q_cc = stm_.q_pos_cc_.segment(0,MODEL_DOF);
    q_dot_virtual_cc = stm_.q_vel_virtual_cc_.segment(0, 6);

//rui - for debug start
    // Eigen::Quaterniond quat_example;
    // Eigen::Vector3d projected_gravity_tester;
    // quat_example.x() = -0.329;
    // quat_example.y() = 0.313;
    // quat_example.z() = 0.845;
    // quat_example.w() = -0.282;
    // projected_gravity_tester = quat_rotate_inverse(quat_example, gravity_vector);
    // std::cout << "projected_gravity_tester" << std::endl;
    // std::cout << projected_gravity_tester << std::endl;
    
    // std::cout << "gravity_vector" << std::endl;
    // std::cout << gravity_vector << std::endl;
    // std::cout << "projected_gravity" << std::endl;
    // std::cout << projected_gravity << std::endl;
//rui - for debug end

//ANCHOR - start of state_cur
    //rui - 2 self.contacts,
    state_cur_(data_idx) = stm_.foot_contact_(0);
    data_idx++;
    state_cur_(data_idx) = stm_.foot_contact_(1);
    data_idx++;
    //rui - 1 self.base_z,
    state_cur_(data_idx) = stm_.base_pos_[2];
    // state_cur_(data_idx) = 0.5;
    data_idx++;

    //rui - 3 self.base_lin_vel * self.obs_scales.lin_vel
    for (auto i = 0; i < 3; i++) 
    {
        // state_cur_(data_idx) = base_lin_vel[i] * obs_scales_lin_vel;
        state_cur_(data_idx) = q_dot_virtual_cc[i] * obs_scales_lin_vel;
        // state_cur_(data_idx) = 0;
        data_idx++;
    }
    //rui - 3 self.base_ang_vel * self.obs_scales.ang_vel,
    for (auto i = 0; i < 3; i++) 
    {
        // state_cur_(data_idx) = base_ang_vel[i] * obs_scales_ang_vel;
        state_cur_(data_idx) = q_dot_virtual_cc[3 + i] * obs_scales_ang_vel;
        // state_cur_(data_idx) = 0;
        data_idx++;
    }

    //rui - 3 self.projected_gravity,
    for (auto i = 0; i < 3; i++) 
    {
        state_cur_(data_idx) = projected_gravity[i];
        // state_cur_(data_idx) = 0;
        // if (i == 2)
        // {
        //     state_cur_(data_idx) = -1;
        // }
        data_idx++;
    }

    //rui - 3 self.commands[:, :3] * self.commands_scale,
    state_cur_(data_idx) = target_vel_x_ * commands_scale_target_vel_x_;
    data_idx++;
    state_cur_(data_idx) = target_vel_y_ * commands_scale_target_vel_y_;
    data_idx++;
    state_cur_(data_idx) = ang_vel_yaw * commands_scale_ang_vel_yaw;
    data_idx++;
    

    //rui - 6 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    for (auto i = 0; i < num_actuator_action; i++) 
    {
        state_cur_(data_idx) = (q_cc(i) - q_init_(i)) * obs_scales_dof_pos;
        data_idx++;
    }

    //rui - 6 self.dof_vel * self.obs_scales.dof_vel,
    for (auto i = 0; i < num_actuator_action; i++) 
    {
        state_cur_(data_idx) = q_dot_cc(i) * obs_scales_dof_vel;
        data_idx++;
    }

    //rui - 6 self.actions
    for (auto i = 0; i < num_actuator_action; i++) 
    {
        state_cur_(data_idx) = rl_action_(i); //DyrosMath::minmax_cut(rl_action_(i), -25.0, 25.0);
        data_idx++;
    }
//ANCHOR - end of state_cur

//ANCHOR - start of csv file
    
    // std::ifstream file("/home/dyros/bolt_ws/src/dyros_bolt_cc/policy_input_3.csv");
    // if (!file.is_open()) {
    //     std::cerr << "Error opening file" << std::endl;
    // }


    // std::string line;
    // int currentLine = 0;
    // linecount_++;

    // while (std::getline(file, line)) {
    //     if (currentLine == linecount_) {
    //         std::stringstream ss(line);
    //         std::string value;
    //         int idx = 0;
    //         while (std::getline(ss, value, ',')) {
    //             state_cur_(idx) = std::stod(value);
    //             idx++;
    //         }
                

    //         //rui - 2 self.contacts,
            
    //         //rui - 1 self.base_z,
            
    //         //rui - 3 self.base_lin_vel * self.obs_scales.lin_vel
            
    //         //rui - 3 self.base_ang_vel * self.obs_scales.ang_vel,

    //         //rui - 3 self.projected_gravity,

    //         //rui - 3 self.commands[:, :3] * self.commands_scale,

    //         //rui - 6 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,

    //         //rui - 6 self.dof_vel * self.obs_scales.dof_vel,

    //         //rui - 6 self.actions
    //     }
    //     currentLine++;
    // }

    // file.close();
//ANCHOR - end of csv file

//ANCHOR - state_cur_output start
    std::ofstream outfile2("/home/dyros/bolt_ws/src/dyros_bolt_cc/state_cur_log.csv", std::ios::app); // Open in append mode
    if (!outfile2.is_open()) {
        std::cerr << "Error opening output file" << std::endl;
    }
    for (int i = 0; i < state_cur_.size(); ++i) 
    {
        outfile2 << state_cur_(i);
        if (i < state_cur_.size() - 1) {
            outfile2 << ", "; // CSV delimiter
        }
    }
    outfile2 << std::endl; // New line for the next vector

//ANCHOR - state_cur_output end

    
    for (int i = 0; i < num_cur_state; i++)
    {
        state_cur_clipped_(i) = DyrosMath::minmax_cut(state_cur_(i), -obs_bound_(i), obs_bound_(i));
    }
    state_.block(0, 0, num_cur_state, 1) = state_cur_clipped_.array();
}

void CustomController::feedforwardPolicy() //rui mlp feedforward
{

    hidden_layer_1 = policy_net_w0_ * state_ + policy_net_b0_;
    hidden_layer_1 = applyELU(hidden_layer_1);

    hidden_layer_2 = policy_net_w2_ * hidden_layer_1 + policy_net_b2_;
    hidden_layer_2 = applyELU(hidden_layer_2);

    hidden_layer_3 = policy_net_w4_ * hidden_layer_2 + policy_net_b4_;
    hidden_layer_3 = applyELU(hidden_layer_3);

    rl_action_ = action_net_w_ * hidden_layer_3 + action_net_b_;

    value_hidden_layer_1 = value_net_w0_ * state_ + value_net_b0_;
    value_hidden_layer_1 = applyELU(value_hidden_layer_1);

    value_hidden_layer_2 = value_net_w2_ * value_hidden_layer_1 + value_net_b2_;
    value_hidden_layer_2 = applyELU(value_hidden_layer_2);

    value_hidden_layer_3 = value_net_w4_ * value_hidden_layer_2 + value_net_b4_;
    value_hidden_layer_3 = applyELU(value_hidden_layer_3);

    value_ = (value_net_w_ * value_hidden_layer_3 + value_net_b_)(0);

//ANCHOR - start of output csv file
    // std::ofstream outfile("/home/dyros/bolt_ws/src/dyros_bolt_cc/output_file_test3.csv", std::ios::app); // Open in append mode
    // if (!outfile.is_open()) {
    //     std::cerr << "Error opening output file" << std::endl;
    // }
    // for (int i = 0; i < rl_action_.size(); ++i) 
    // {
    //     outfile << rl_action_(i);
    //     if (i < rl_action_.size() - 1) {
    //         outfile << ", "; // CSV delimiter
    //     }
    // }
    // outfile << std::endl; // New line for the next vector
//ANCHOR - end of output csv file

}

void CustomController::computeSlow() //rui main
{
    copyRobotData(rd_);
    if (rd_cc_.tc_.mode == 8)
    {
        if (rd_cc_.tc_init)
        {
            start_time_ = rd_cc_.control_time_us_;
            std::cout<<"cc mode 8"<<std::endl;
        
            torque_init_ = rd_cc_.torque_desired;

            processNoise();
            processObservation();
            std::cout << "state_cur" << std::endl;
            std::cout << state_cur_ << std::endl;
            state_.block(0,0, num_cur_state, 1) = state_cur_.array();

            rd_.tc_init = false;
        }

        processNoise();

        // processObservation and feedforwardPolicy mean time: 15 us, max 53 us 
        if (framecounter_ == frameskip) //rui 250hz 변수만들어서 바꿔주기
        {
            processObservation();
            feedforwardPolicy();
            
            framecounter_ = 0; 
        }
        framecounter_++;

        for (int i = 0; i < num_actuator_action; i++)
        {
            torque_rl_(i) = DyrosMath::minmax_cut(rl_action_(i), -torque_bound_(i), torque_bound_(i));
        }
        if (rd_cc_.control_time_us_ < start_time_ + 0.2e6) //rui torque 쏴주는것
        {
            for (int i = 0; i <MODEL_DOF; i++)
            {
                torque_spline_(i) = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 0.2e6, torque_init_(i), torque_rl_(i), 0.0, 0.0);
            }
            rd_.torque_desired = torque_spline_;
        }
        else
        {
             rd_.torque_desired = torque_rl_;
        }
        
        rd_.torque_desired.setZero();
        rd_.torque_desired[5] = 0.1;

//ANCHOR - start of output csv file
        // std::ofstream outfile3("/home/dyros/bolt_ws/src/dyros_bolt_cc/output_file_test_3.csv", std::ios::app); // Open in append mode
        // if (!outfile3.is_open()) {
        //     std::cerr << "Error opening output file" << std::endl;
        // }
        // for (int i = 0; i < torque_rl_.size(); ++i) 
        // {
        //     outfile3 << torque_rl_(i);
        //     if (i < torque_rl_.size() - 1) {
        //         outfile3 << ", "; // CSV delimiter
        //     }
        // }
        // outfile3 << std::endl; // New line for the next vector
//ANCHOR - end of output csv file

        

//ANCHOR - for debug start
        // std::cout << "torque_rl_" << std::endl;
        // std::cout << torque_rl_ << std::endl;
        // std::cout << "rl_action_" << std::endl;
        // std::cout << rl_action_ << std::endl;
        // try {
        //     YAML::Node config = YAML::LoadFile("/home/dyros/bolt_ws/src/dyros_bolt_cc/src/torque.yaml");  // Load the YAML file
        //     Eigen::VectorXd torque_vector(6);

        //         for (int i = 0; i < 6; ++i) {
        //             torque_vector(i) = config["torque"][i].as<double>();  // Fill the vector
        //     }

        //     rd_.torque_desired = torque_vector.transpose();
        // } catch (const YAML::Exception& e) {
        //     std::cerr << "Error reading YAML file: " << e.what() << std::endl;
        // }
        // std::cout << "rd_.torque_desired" << std::endl;
        // std::cout << rd_.torque_desired << std::endl;
//ANCHOR - for debug end


        // if (value_ < 50.0)
        // {
        //     if (stop_by_value_thres_ == false)
        //     {
        //         stop_by_value_thres_ = true;
        //         stop_start_time_ = rd_cc_.control_time_us_;
        //         q_stop_ = q_noise_;
        //         std::cout << "Stop by Value Function" << std::endl;
        //     }
        // }
        // if (stop_by_value_thres_)
        // {
        //     rd_.torque_desired = kp_ * (q_stop_ - q_noise_) - kv_*q_vel_noise_;
        // }

        // if (is_write_file_) //rui 파일 write
        // {
        //     if ((rd_cc_.control_time_us_ - time_write_pre_)/1e6 > 1/240.0)
        //     {
        //         writeFile << (rd_cc_.control_time_us_ - start_time_)/1e6 << "\t";
        //         // writeFile << phase_ << "\t";
        //         // writeFile << DyrosMath::minmax_cut(rl_action_(num_action-1)*1/policy_frequency_, 0.0, 1/policy_frequency_) << "\t";

        //         writeFile << rd_cc_.LF_FT.transpose() << "\t";
        //         writeFile << rd_cc_.RF_FT.transpose() << "\t";
        //         writeFile << rd_cc_.LF_CF_FT.transpose() << "\t";
        //         writeFile << rd_cc_.RF_CF_FT.transpose() << "\t";

        //         writeFile << rd_cc_.torque_desired.transpose()  << "\t";
        //         writeFile << q_noise_.transpose() << "\t";
        //         writeFile << q_dot_lpf_.transpose() << "\t";
        //         writeFile << rd_cc_.q_dot_virtual_.transpose() << "\t";
        //         writeFile << rd_cc_.q_virtual_.transpose() << "\t";

        //         writeFile << value_ << "\t" << stop_by_value_thres_;
               
        //         writeFile << std::endl;

        //         time_write_pre_ = rd_cc_.control_time_us_;
        //     }
        // }
    }
}

void CustomController::computeFast()
{
    // if (tc.mode == 10)
    // {
    // }
    // else if (tc.mode == 11)
    // {
    // }
}

void CustomController::computePlanner()
{
}

void CustomController::copyRobotData(RobotData &rd_l)
{
    std::memcpy(&rd_cc_, &rd_l, sizeof(RobotData));
}

void CustomController::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    target_vel_x_ = DyrosMath::minmax_cut(0.5*joy->axes[1], -0.2, 0.5);
    target_vel_y_ = DyrosMath::minmax_cut(0.5*joy->axes[0], -0.2, 0.2);
}

