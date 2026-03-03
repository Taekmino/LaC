#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32MultiArray.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/PointCloud2.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unordered_map>
#include <tuple>
#include <cmath>
#include <string>

// Hash function for 2D grid cell keys represented as std::tuple<int,int>.
struct HashTuple2D {
  std::size_t operator()(const std::tuple<int,int> &key) const {
    auto h1 = std::hash<int>()(std::get<0>(key));
    auto h2 = std::hash<int>()(std::get<1>(key));
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

// Hash function for 3D voxel keys represented as std::tuple<int,int,int>.
struct HashTuple
{
    size_t operator()(const std::tuple<int,int,int>& key) const
    {
        auto h1 = std::hash<int>()(std::get<0>(key));
        auto h2 = std::hash<int>()(std::get<1>(key));
        auto h3 = std::hash<int>()(std::get<2>(key));
        return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1);
    }
};

// Dual-channel mapping node that generates both an obstacle map (from depth) and
// a hazard cost map (from segmentation masks with anxiety scores). The two maps
// are fused via max-fusion into a final occupancy grid for navigation planning.
class TwoDObstacleCostMapNode
{
public:
  TwoDObstacleCostMapNode(ros::NodeHandle& nh)
  : nh_(nh)
  {
    // Load parameters
    nh_.param("z_max_threshold", z_max_threshold_, 1.3f);
    nh_.param("z_min_threshold", z_min_threshold_, -0.1f);
    nh_.param("min_vis_threshold", min_vis_threshold_, 0.1f);
    nh_.param("max_cost_depth_threshold", max_cost_depth_threshold_, 3.0f);
    nh_.param("max_obstacle_depth_threshold", max_obstacle_depth_threshold_, 3.0f);
    nh_.param("grid_resolution", grid_res_, 0.05f);
    nh_.param("initial_grid_size_x", grid_size_x_, 200);
    nh_.param("initial_grid_size_y", grid_size_y_, 200);
    nh_.param("depth_factor", depth_factor_, 1000.0f);

    // Camera intrinsic parameters (ZED stereo camera)
    nh_.param("fx", fx_, 476.4738387248956f);
    nh_.param("fy", fy_, 476.4738387248956f);
    nh_.param("cx", cx_, 336.3356590270996f);
    nh_.param("cy", cy_, 191.4279479980469f);

    nh_.param("sliding_window_size", sliding_window_size_, 6);

    // Sigma computation parameters
    nh_.param("sigma_method", sigma_method_, 0); // 0: fixed, 1: log-based
    nh_.param("T_param", T_param_, 1.0f);        // T parameter for log-based method
    nh_.param("sigma_k_1", sigma_k_1_, 0.15f);   // base sigma for anxiety_score = 1
    nh_.param("sigma_k_2", sigma_k_2_, 0.2f);    // base sigma for anxiety_score = 2
    nh_.param("sigma_k_3", sigma_k_3_, 0.25f);   // base sigma for anxiety_score = 3

    // Subscribers for obstacle map pipeline (depth + odometry)
    depth_sub_for_obs_.subscribe(nh_, "/viver1/front/depth", 1);
    odom_sub_for_obs_.subscribe(nh_, "/viver1/odometry", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry> ObsSyncPolicy;
    obstacle_sync_.reset(new message_filters::Synchronizer<ObsSyncPolicy>(ObsSyncPolicy(10), depth_sub_for_obs_, odom_sub_for_obs_));
    obstacle_sync_->registerCallback(boost::bind(&TwoDObstacleCostMapNode::obstacleMapCallback, this, _1, _2));

    // Subscribers for cost map pipeline (depth + odometry + segmentation mask)
    depth_sub_for_cost_.subscribe(nh_, "/viver1/front/depth", 200);
    odom_sub_for_cost_.subscribe(nh_, "/viver1/odometry", 200);
    seg_sub_.subscribe(nh_, "/segmentation_mask", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry, sensor_msgs::Image> CostSyncPolicy;
    costmap_sync_.reset(new message_filters::Synchronizer<CostSyncPolicy>(CostSyncPolicy(500), depth_sub_for_cost_, odom_sub_for_cost_, seg_sub_));
    costmap_sync_->registerCallback(boost::bind(&TwoDObstacleCostMapNode::costMapCallback, this, _1, _2, _3));

    // Publishers
    final_map_pub_      = nh_.advertise<sensor_msgs::PointCloud2>("final_map_cloud", 1);
    occupancy_grid_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("obstacle_map", 1);
    occ_grid_pub_       = nh_.advertise<nav_msgs::OccupancyGrid>("final_cost_grid", 1);

    ROS_INFO("TwoDObstacleCostMapNode initialized.");
  }

private:
  // Removes noisy depth edges from a 16-bit depth image using Canny edge detection.
  // Edge pixels are invalidated (set to UINT16_MAX) so they are skipped during 3D projection.
  void filterDepthEdge(cv::Mat& depth_img, float depthrange)
  {
      // Clamp depth to (depthrange + 1) meters in millimeters
      cv::Mat depth_img_range_norm = depth_img.clone();
      double maxVal = (depthrange + 1.0) * 1000.0;
      cv::min(depth_img_range_norm, maxVal, depth_img_range_norm);

      // Convert to 8-bit for Canny edge detection
      depth_img_range_norm.convertTo(depth_img_range_norm, CV_8U,
                                    255.0 / maxVal, 0.0);

      // Detect edges using Canny
      cv::Mat edges;
      cv::Canny(depth_img_range_norm, edges, 50, 150);

      // Dilate edges to cover adjacent boundary pixels
      cv::Mat kernel_edges = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
      cv::dilate(edges, edges, kernel_edges);

      // Invalidate depth at edge locations
      depth_img.setTo(std::numeric_limits<uint16_t>::max(), edges);
  }

  // Obstacle map callback: builds a 2D obstacle occupancy grid from depth + odometry.
  // Uses a sliding window to accumulate point clouds, then projects to a 2D grid
  // filtering by z-height range.
  void obstacleMapCallback(const sensor_msgs::ImageConstPtr& depth_msg,
                           const nav_msgs::OdometryConstPtr& odom_msg)
  {
    ROS_INFO("[ObstacleMap] Received depth+odom. Building/Updating 2D obstacle map...");

    latest_odom_.header = odom_msg->header;
    latest_odom_.pose = odom_msg->pose;
    latest_odom_.twist = odom_msg->twist;

    // Convert depth image
    cv_bridge::CvImageConstPtr cv_depth_ptr;
    try {
      cv_depth_ptr = cv_bridge::toCvShare(depth_msg);
    } catch (const cv_bridge::Exception &e) {
      ROS_WARN("[ObstacleMap] Depth cv_bridge exception: %s", e.what());
      return;
    }

    // Apply edge filter for 16-bit depth images
    cv::Mat processed_depth;
    if (cv_depth_ptr->image.type() == CV_16UC1) {
      processed_depth = cv_depth_ptr->image.clone();
      filterDepthEdge(processed_depth, max_obstacle_depth_threshold_);
    } else {
      processed_depth = cv_depth_ptr->image;
    }

    // Generate point cloud from current frame
    processDepthImage(processed_depth);

    // Update sliding window
    sliding_window_obstacle_.push_back(current_cloud_);
    if (sliding_window_obstacle_.size() > sliding_window_size_) {
      sliding_window_obstacle_.pop_front();
    }

    // Build local 2D obstacle map once enough frames are accumulated
    std::unordered_map<std::tuple<int,int>, bool, HashTuple2D> local_obstacle_map;
    if (sliding_window_obstacle_.size() == sliding_window_size_) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
      for (const auto &cloud : sliding_window_obstacle_) {
        *combined_cloud += *cloud;
      }

      // Project points within valid z-range to 2D grid cells
      for (const auto &point : combined_cloud->points)
      {
        if (point.z < z_min_threshold_ || point.z > z_max_threshold_) {
          continue;
        }
        int vx = static_cast<int>(std::floor(point.x / grid_res_));
        int vy = static_cast<int>(std::floor(point.y / grid_res_));
        std::tuple<int,int> cell(vx, vy);
        local_obstacle_map[cell] = true;
      }
      // Accumulate local obstacles into global map
      for (const auto &kv : local_obstacle_map) {
        global_obstacle_map_[kv.first] = true;
      }
    }

    // Publish global obstacle occupancy grid
    publishGlobalOccupancyGrid();

    updateFinalMap();
    if (!final_map_.empty()) {
      publishFinalOccupancyGrid();
      publishFinalMapPointCloud(depth_msg->header.stamp);
    }
  }

private:
  // Converts a depth image to a global-frame colored point cloud using the latest odometry.
  // Back-projects depth pixels to 3D, applies camera-to-base and base-to-global transforms.
  void processDepthImage(const cv::Mat &depth_img)
  {
    const auto &odom_msg = latest_odom_;

    // Build global transform from odometry
    double tx = odom_msg.pose.pose.position.x;
    double ty = odom_msg.pose.pose.position.y;
    double tz = odom_msg.pose.pose.position.z;
    double qx = odom_msg.pose.pose.orientation.x;
    double qy = odom_msg.pose.pose.orientation.y;
    double qz = odom_msg.pose.pose.orientation.z;
    double qw = odom_msg.pose.pose.orientation.w;

    Eigen::Matrix4f global_tf = Eigen::Matrix4f::Identity();
    {
      Eigen::Quaternionf quat((float)qw, (float)qx, (float)qy, (float)qz);
      Eigen::Matrix3f rot3 = quat.toRotationMatrix();
      global_tf.block<3,3>(0,0) = rot3;
      global_tf(0,3) = (float)tx;
      global_tf(1,3) = (float)ty;
      global_tf(2,3) = (float)tz;
    }

    int width = depth_img.cols;
    int height = depth_img.rows;
    current_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    current_cloud_->points.reserve(width * height);

    for (int v = 0; v < height; v++){
      for (int u = 0; u < width; u++){
        float raw_depth = 0.0f;
        if(depth_img.type() == CV_16UC1){
          uint16_t d_raw = depth_img.at<uint16_t>(v, u);
          if (d_raw == std::numeric_limits<uint16_t>::max())
            continue;
          raw_depth = static_cast<float>(d_raw) / depth_factor_;
        } else if(depth_img.type() == CV_32FC1){
          float d_raw = depth_img.at<float>(v, u);
          raw_depth = d_raw / depth_factor_;
        }
        if (raw_depth < 0.0001f || raw_depth > max_obstacle_depth_threshold_)
          continue;

        // Back-project to camera coordinates
        float Z = raw_depth;
        float X = (u - cx_) * Z / fx_;
        float Y = (v - cy_) * Z / fy_;

        // Camera-to-base transform: Z->X, X->-Y, Y->-Z
        Eigen::Vector4f cam_pt(X, Y, Z, 1.0f);
        Eigen::Matrix4f T_cam_to_base = Eigen::Matrix4f::Identity();
        Eigen::Matrix3f R_cam_to_base;
        R_cam_to_base <<  0,  0,  1,
                         -1,  0,  0,
                          0, -1,  0;
        T_cam_to_base.block<3,3>(0,0) = R_cam_to_base;
        cam_pt = T_cam_to_base * cam_pt;

        // Transform to global coordinates
        Eigen::Vector4f global_pt = global_tf * cam_pt;

        pcl::PointXYZRGB pcl_pt;
        pcl_pt.x = global_pt.x();
        pcl_pt.y = global_pt.y();
        pcl_pt.z = global_pt.z();
        pcl_pt.r = 255;
        pcl_pt.g = 255;
        pcl_pt.b = 255;
        current_cloud_->points.push_back(pcl_pt);
      }
    }
  }

  // Publishes the global obstacle map as a ROS OccupancyGrid message.
  // Computes the bounding box of all occupied cells and encodes them in a grid
  // with 0 = free and 100 = occupied.
  void publishGlobalOccupancyGrid()
  {
    if (global_obstacle_map_.empty()) {
      ROS_WARN("[ObstacleMap] Global obstacle map is empty, not publishing occupancy grid.");
      return;
    }

    // Compute bounding box of occupied cells
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();

    for (const auto &kv : global_obstacle_map_) {
      int x = std::get<0>(kv.first);
      int y = std::get<1>(kv.first);
      if (x < min_x) min_x = x;
      if (y < min_y) min_y = y;
      if (x > max_x) max_x = x;
      if (y > max_y) max_y = y;
    }

    int width  = max_x - min_x + 1;
    int height = max_y - min_y + 1;

    nav_msgs::OccupancyGrid grid_msg;
    grid_msg.header.stamp = ros::Time::now();
    grid_msg.header.frame_id = "odom";
    grid_msg.info.resolution = grid_res_;
    grid_msg.info.width = width;
    grid_msg.info.height = height;
    grid_msg.info.origin.position.x = min_x * grid_res_;
    grid_msg.info.origin.position.y = min_y * grid_res_;
    grid_msg.info.origin.position.z = 0.0;
    grid_msg.info.origin.orientation.w = 1.0;
    grid_msg.info.origin.orientation.x = 0.0;
    grid_msg.info.origin.orientation.y = 0.0;
    grid_msg.info.origin.orientation.z = 0.0;

    // Initialize all cells as free (0), then mark occupied cells as 100
    grid_msg.data.resize(width * height, 0);
    for (const auto &kv : global_obstacle_map_) {
      int x = std::get<0>(kv.first);
      int y = std::get<1>(kv.first);
      int grid_x = x - min_x;
      int grid_y = y - min_y;
      int index = grid_y * width + grid_x;
      grid_msg.data[index] = 100;
    }

    occupancy_grid_pub_.publish(grid_msg);
  }

  // Cost map callback: builds a hazard cost map from synchronized depth + odometry +
  // segmentation mask. Segmentation pixels encode anxiety scores (251 = detected but
  // unscored, 253-255 = anxiety levels 1-3). Uses a sliding window with voxel-based
  // filtering to require consistent detections across at least 3 frames.
  void costMapCallback(const sensor_msgs::ImageConstPtr& depth_msg,
                       const nav_msgs::OdometryConstPtr& odom_msg,
                       const sensor_msgs::ImageConstPtr& seg_msg)
  {
    ROS_INFO("[CostMap] Received depth+odom+seg. Building 2D costmap...");

    // Convert depth image (clone for modification)
    cv_bridge::CvImageConstPtr cv_depth_ptr;
    try {
      cv_depth_ptr = cv_bridge::toCvShare(depth_msg);
    } catch(const cv_bridge::Exception &e){
      ROS_WARN("[CostMap] Depth cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat depth_img = cv_depth_ptr->image.clone();
    if(depth_img.empty()) {
      ROS_WARN("[CostMap] Empty depth image");
      return;
    }

    // Filter out depth edges (boundary artifact removal)
    filterDepthEdge(depth_img, max_cost_depth_threshold_);

    // Convert segmentation mask
    cv_bridge::CvImageConstPtr cv_seg_ptr;
    try {
      cv_seg_ptr = cv_bridge::toCvShare(seg_msg, "mono8");
    } catch(const cv_bridge::Exception &e){
      ROS_WARN("[CostMap] Segmentation cv_bridge exception: %s", e.what());
      return;
    }

    // Build global transform from odometry
    double tx = odom_msg->pose.pose.position.x;
    double ty = odom_msg->pose.pose.position.y;
    double tz = odom_msg->pose.pose.position.z;
    double qx = odom_msg->pose.pose.orientation.x;
    double qy = odom_msg->pose.pose.orientation.y;
    double qz = odom_msg->pose.pose.orientation.z;
    double qw = odom_msg->pose.pose.orientation.w;

    Eigen::Matrix4f global_tf = Eigen::Matrix4f::Identity();
    {
      Eigen::Quaternionf quat((float)qw, (float)qx, (float)qy, (float)qz);
      Eigen::Matrix3f rot3 = quat.toRotationMatrix();
      global_tf.block<3,3>(0,0) = rot3;
      global_tf(0,3) = (float)tx;
      global_tf(1,3) = (float)ty;
      global_tf(2,3) = (float)tz;
    }

    // Build point cloud with anxiety intensity from depth + segmentation
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZI>());

    int width  = depth_img.cols;
    int height = depth_img.rows;
    current_cloud->reserve(width * height);

    for(int v = 0; v < height; v++){
      for(int u = 0; u < width; u++){
        float raw_depth = 0.0f;

        // Skip invalidated edge pixels
        if(depth_img.type() == CV_16UC1) {
          uint16_t d_raw = depth_img.at<uint16_t>(v, u);
          if(d_raw == std::numeric_limits<uint16_t>::max()) {
            continue;
          }
          raw_depth = static_cast<float>(d_raw) / depth_factor_;
        }
        else if(depth_img.type() == CV_32FC1) {
          float d_raw = depth_img.at<float>(v, u);
          raw_depth = d_raw / depth_factor_;
        }

        if(raw_depth < 0.0001f || raw_depth > max_cost_depth_threshold_) {
          continue;
        }

        // Read segmentation value encoding the anxiety level
        uint8_t seg_val = cv_seg_ptr->image.at<uint8_t>(v, u);

        // Determine anxiety score from segmentation pixel value
        float anxiety_score = 0.0f;
        bool assigned = false;

        if(seg_val == 251){
          anxiety_score = 1.0f;  // Detected but anxiety score not yet assigned
          assigned = false;
        }
        else if(seg_val >= 253 && seg_val <= 255){
          anxiety_score = float(seg_val - 252);  // 253->1, 254->2, 255->3
          assigned = true;
        }
        else {
          // Skip non-hazard pixels
          continue;
        }

        // Back-project to camera coordinates
        float Z = raw_depth;
        float X = (u - cx_) * Z / fx_;
        float Y = (v - cy_) * Z / fy_;
        Eigen::Vector4f cam_pt(X, Y, Z, 1.0f);

        // Camera-to-base transform: Z->X, X->-Y, Y->-Z
        Eigen::Matrix4f T_cam_to_base = Eigen::Matrix4f::Identity();
        Eigen::Matrix3f R_cam_to_base;
        R_cam_to_base <<  0,  0,  1,
                        -1,  0,  0,
                          0, -1,  0;
        T_cam_to_base.block<3,3>(0,0) = R_cam_to_base;
        cam_pt = T_cam_to_base * cam_pt;

        // Transform to global coordinates
        Eigen::Vector4f global_pt = global_tf * cam_pt;

        // Store point with anxiety score as intensity
        pcl::PointXYZI pcl_pt;
        pcl_pt.x = global_pt.x();
        pcl_pt.y = global_pt.y();
        pcl_pt.z = global_pt.z();
        pcl_pt.intensity = anxiety_score;

        current_cloud->points.push_back(pcl_pt);
      }
    }

    // Update sliding window for temporal consistency filtering
    sliding_window_cost_.push_back(current_cloud);
    if (sliding_window_cost_.size() > sliding_window_size_) {
        sliding_window_cost_.pop_front();
    }

    // When the window is full, apply voxel-based consistency check:
    // only voxels observed in at least 3 frames are kept in the cost map.
    if (sliding_window_cost_.size() == sliding_window_size_) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr combined_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        for (const auto &cloud_in_window : sliding_window_cost_) {
            *combined_cloud += *cloud_in_window;
        }

      // Count observations per voxel and track maximum anxiety score
      std::unordered_map<std::tuple<int,int,int>, int, HashTuple> voxel_count;
      std::unordered_map<std::tuple<int,int,int>, float, HashTuple> voxel_anxiety;

      for(const auto &pt : combined_cloud->points){
        float z = pt.z;
        if (z < z_min_threshold_ || z > z_max_threshold_) {
          continue;
        }

        int vx = static_cast<int>(std::floor(pt.x / grid_res_));
        int vy = static_cast<int>(std::floor(pt.y / grid_res_));
        int vz = static_cast<int>(std::floor(pt.z / grid_res_));
        auto voxel_key = std::make_tuple(vx, vy, vz);

        voxel_count[voxel_key]++;
        float current_max = 0.0f;
        if(voxel_anxiety.find(voxel_key) != voxel_anxiety.end()){
          current_max = voxel_anxiety[voxel_key];
        }
        voxel_anxiety[voxel_key] = std::max(current_max, pt.intensity);
      }

      // Only keep voxels observed at least 3 times for noise robustness
      for(const auto &kv : voxel_count){
        int count = kv.second;
        if(count < 3) {
          continue;
        }

        const auto &voxel_key = kv.first;
        float anxiety_score = voxel_anxiety[voxel_key];

        // Project to 2D grid cell
        int gx = std::get<0>(voxel_key);
        int gy = std::get<1>(voxel_key);
        std::tuple<int,int> cell_key(gx, gy);

        // Keep the maximum anxiety score for each cell
        float old_score = 0.0f;
        auto it = cost_map_.find(cell_key);
        if(it != cost_map_.end()){
          old_score = it->second;
        }
        cost_map_[cell_key] = std::max(old_score, anxiety_score);
      }
    }

    // Update and publish the fused final map
    updateFinalMap();
    publishFinalMapPointCloud(depth_msg->header.stamp);
    publishFinalOccupancyGrid();

    ROS_INFO("[CostMap] Cost map updated. Current cell count: %zu", cost_map_.size());
  }

  // Fuses the global obstacle map and the anxiety cost map into a final navigation map.
  // Obstacle cells are set to cost 100. Cost map cells are spread using 2D Gaussian
  // propagation based on anxiety scores (max-fusion strategy).
  void updateFinalMap()
  {
    final_map_.clear();

    // Mark all obstacle cells as fully occupied (cost = 100).
    // The Gaussian spread loop below skips cells already at 100 to avoid overwriting obstacles.
    for (const auto &kv : global_obstacle_map_)
    {
      final_map_[kv.first] = 100.0f;
    }

    // Apply Gaussian spread for each anxiety-scored cell
    for (const auto &kv : cost_map_)
    {
      float anxiety_score = kv.second;
      if (anxiety_score <= 0.0f)
        continue;

      float sigma = computeSigma(anxiety_score);
      int gx_center = std::get<0>(kv.first);
      int gy_center = std::get<1>(kv.first);

      float center_x = gx_center * grid_res_ + 0.5f * grid_res_;
      float center_y = gy_center * grid_res_ + 0.5f * grid_res_;

      // Spread cost within 3-sigma radius
      int max_radius_cell = static_cast<int>(std::ceil(3.0f * sigma / grid_res_));

      for (int dx = -max_radius_cell; dx <= max_radius_cell; dx++)
      {
        for (int dy = -max_radius_cell; dy <= max_radius_cell; dy++)
        {
          int gx = gx_center + dx;
          int gy = gy_center + dy;

          float nx = gx * grid_res_ + 0.5f * grid_res_;
          float ny = gy * grid_res_ + 0.5f * grid_res_;

          float dist_x = nx - center_x;
          float dist_y = ny - center_y;
          float dist_sq = dist_x * dist_x + dist_y * dist_y;

          // Gaussian cost: exp(-0.5 * d^2 / sigma^2) * (anxiety / 3) * 100
          float exponent = -0.5f * (dist_sq / (sigma * sigma));
          float gauss_val = std::exp(exponent);
          float cost_from_gauss = gauss_val * (anxiety_score / 3.0f) * 100.0f;

          // Max-fusion: keep the highest cost, but do not override obstacles (cost 100)
          float &prev_val = final_map_[std::make_tuple(gx, gy)];
          if (prev_val < 100.0f)
          {
            prev_val = std::max(prev_val, cost_from_gauss);
            if (prev_val > 100.0f)
              prev_val = 100.0f;
          }
        }
      }
    }
  }

  // Publishes the final fused map as a PointCloud2 message for visualization.
  // The z-coordinate encodes cost height and intensity encodes cost level.
  void publishFinalMapPointCloud(const ros::Time& stamp)
  {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    cloud->header.frame_id = "odom";
    cloud->points.reserve(final_map_.size());

    for(const auto &kv : final_map_)
    {
      const auto &cell = kv.first;
      float cost_val = kv.second;

      if(cost_val < min_vis_threshold_)
        continue;

      float gx = static_cast<float>(std::get<0>(cell)) * grid_res_ + 0.5f * grid_res_;
      float gy = static_cast<float>(std::get<1>(cell)) * grid_res_ + 0.5f * grid_res_;

      pcl::PointXYZI pt;
      pt.x = gx;
      pt.y = gy;
      pt.z = (cost_val) * 0.010f;  // Map cost 0-100 to height 0-1m
      pt.intensity = 100 - cost_val;

      cloud->points.push_back(pt);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header.stamp = stamp;
    final_map_pub_.publish(output);

    ROS_INFO("[FinalMap] Published final map pointcloud (0~100 cost + obstacles).");
  }

  // Publishes the final fused map as a ROS OccupancyGrid with cost values 0-100.
  void publishFinalOccupancyGrid()
  {
    if(final_map_.empty()){
      ROS_WARN("[CostMap] Final map is empty, not publishing occupancy grid.");
      return;
    }

    // Compute bounding box
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();

    for(const auto &kv : final_map_){
      int gx = std::get<0>(kv.first);
      int gy = std::get<1>(kv.first);
      if(gx < min_x) min_x = gx;
      if(gy < min_y) min_y = gy;
      if(gx > max_x) max_x = gx;
      if(gy > max_y) max_y = gy;
    }

    int width  = max_x - min_x + 1;
    int height = max_y - min_y + 1;

    nav_msgs::OccupancyGrid occ_grid;
    occ_grid.header.stamp = ros::Time::now();
    occ_grid.header.frame_id = "odom";
    occ_grid.info.resolution = grid_res_;
    occ_grid.info.width  = width;
    occ_grid.info.height = height;

    occ_grid.info.origin.position.x = min_x * grid_res_;
    occ_grid.info.origin.position.y = min_y * grid_res_;
    occ_grid.info.origin.position.z = 0.0;
    occ_grid.info.origin.orientation.w = 1.0;
    occ_grid.info.origin.orientation.x = 0.0;
    occ_grid.info.origin.orientation.y = 0.0;
    occ_grid.info.origin.orientation.z = 0.0;

    // Initialize all cells as free, then fill in costs
    occ_grid.data.resize(width * height, 0);

    for(const auto &kv : final_map_){
      int gx = std::get<0>(kv.first);
      int gy = std::get<1>(kv.first);
      int index = (gy - min_y) * width + (gx - min_x);

      int cost = static_cast<int>(kv.second);
      if(cost > 100) cost = 100;
      if(cost < 0)   cost = 0;

      occ_grid.data[index] = cost;
    }

    occ_grid_pub_.publish(occ_grid);
    ROS_INFO("[CostMap] Published final occupancy grid (size: %d x %d)", width, height);
  }


private:
  ros::NodeHandle nh_;

  // Parameters
  float z_max_threshold_;
  float z_min_threshold_;
  float max_cost_depth_threshold_;
  float max_obstacle_depth_threshold_;
  float grid_res_;
  int   grid_size_x_;
  int   grid_size_y_;
  float depth_factor_;
  float fx_, fy_, cx_, cy_;
  float min_vis_threshold_;
  int sliding_window_size_;

  // Sigma computation parameters
  int   sigma_method_;
  float T_param_;
  float sigma_k_1_, sigma_k_2_, sigma_k_3_;

  std::deque<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> sliding_window_obstacle_;
  std::deque<pcl::PointCloud<pcl::PointXYZI>::Ptr> sliding_window_cost_;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud_;

  nav_msgs::Odometry latest_odom_;

  // 2D obstacle map: key = (gx, gy), value = obstacle presence
  std::unordered_map<std::tuple<int,int>, bool, HashTuple2D> obstacle_map_;
  std::unordered_map<std::tuple<int,int>, bool, HashTuple2D> global_obstacle_map_;

  // 2D cost map: key = (gx, gy), value = maximum anxiety score
  std::unordered_map<std::tuple<int,int>, float, HashTuple2D> cost_map_;

  // Final fused map (obstacles + Gaussian-spread anxiety costs)
  std::unordered_map<std::tuple<int,int>, float, HashTuple2D> final_map_;

  // Computes the Gaussian spread sigma based on the anxiety score.
  // Method 0 (fixed): returns a fixed per-level sigma (sigma_k_1/2/3).
  // Method 1 (log-based): sigma = sigma_k * log(anxiety_score / T_param).
  //   Falls back to sigma_k when the log factor would produce a value < 0.01.
  inline float computeSigma(float anxiety_score)
  {
    float sigma_k;
    if      (anxiety_score <= 1.0f) sigma_k = sigma_k_1_;
    else if (anxiety_score <= 2.0f) sigma_k = sigma_k_2_;
    else                            sigma_k = sigma_k_3_;

    if (sigma_method_ == 1)
    {
      if (anxiety_score <= 0.0f || T_param_ <= 0.0f)
        return sigma_k;

      float ratio = anxiety_score / T_param_;
      if (ratio <= 0.0f)
        return sigma_k;

      float sigma_new = sigma_k * std::log(ratio);
      return (sigma_new < 0.01f) ? 0.01f : sigma_new;
    }

    // Method 0 (default): fixed sigma per anxiety level
    return sigma_k;
  }

  // Subscribers for obstacle map pipeline
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_for_obs_;
  message_filters::Subscriber<nav_msgs::Odometry> odom_sub_for_obs_;
  std::shared_ptr<message_filters::Synchronizer<
    message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>>> obstacle_sync_;

  // Subscribers for cost map pipeline
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_for_cost_;
  message_filters::Subscriber<nav_msgs::Odometry> odom_sub_for_cost_;
  message_filters::Subscriber<sensor_msgs::Image> seg_sub_;
  std::shared_ptr<message_filters::Synchronizer<
    message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry, sensor_msgs::Image>>> costmap_sync_;

  // Publishers
  ros::Publisher final_map_pub_;
  ros::Publisher occupancy_grid_pub_;
  ros::Publisher occ_grid_pub_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "two_d_obstacle_cost_map_node");
  ros::NodeHandle nh("~");
  TwoDObstacleCostMapNode node(nh);
  ros::spin();
  return 0;
}
