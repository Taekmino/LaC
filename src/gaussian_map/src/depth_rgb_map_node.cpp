#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

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
#include <deque>
#include <cmath>
#include <string>
#include <limits>

// Hash function for 3D voxel keys represented as std::tuple<int,int,int>.
struct HashTuple {
  std::size_t operator()(const std::tuple<int,int,int> &key) const {
    auto h1 = std::hash<int>()(std::get<0>(key));
    auto h2 = std::hash<int>()(std::get<1>(key));
    auto h3 = std::hash<int>()(std::get<2>(key));
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

// Removes noisy depth edges from a 16-bit depth image.
// Normalizes depth to 8-bit, applies Canny edge detection, dilates the edges,
// and invalidates (sets to UINT16_MAX) pixels near depth discontinuities.
void filterDepthEdge(cv::Mat &depth_img, float depthrange)
{
  if (depth_img.empty() || depth_img.type() != CV_16UC1) {
    return;
  }

  // Clone and clamp depth values to (depthrange + 1) meters in millimeters
  cv::Mat depth_img_range_norm = depth_img.clone();
  float maxDepthMillimeters = (depthrange + 1.0f) * 1000.0f;
  cv::min(depth_img_range_norm, maxDepthMillimeters, depth_img_range_norm);

  // Normalize to 8-bit range [0, 255]
  depth_img_range_norm.convertTo(depth_img_range_norm,
                                 CV_8U,
                                 255.0 / maxDepthMillimeters);

  // Detect edges using Canny
  cv::Mat edges;
  cv::Canny(depth_img_range_norm, edges, 50, 150);

  // Dilate edges to cover adjacent noisy pixels
  cv::Mat kernel_edges = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::dilate(edges, edges, kernel_edges);

  // Invalidate depth at edge locations
  depth_img.setTo(std::numeric_limits<uint16_t>::max(), edges);
}

// Builds a voxelized RGB-D map from synchronized depth, color, and odometry data.
// Maintains a sliding window of recent frames and publishes both local and global
// voxel maps as colored point clouds.
class DepthRGBMapNode
{
public:
  DepthRGBMapNode(ros::NodeHandle& nh)
  : nh_(nh)
  {
    // Camera intrinsic parameters (ZED stereo camera)
    nh_.param("fx", fx_, 476.4738f);
    nh_.param("fy", fy_, 476.4738f);
    nh_.param("cx", cx_, 336.3356f);
    nh_.param("cy", cy_, 191.4279f);
    nh_.param("depth_factor", factor_, 1000.0f);
    nh_.param("voxel_size", voxel_size_, 0.05f);
    nh_.param("max_range", max_range_, 3.4f);
    nh_.param("apply_cam_to_base", apply_cam_to_base_, true);
    // Maximum z-coordinate for global map accumulation
    nh_.param("max_z_threshold", max_z_threshold_, 1.5f);

    // Subscribers
    depth_sub_.subscribe(nh_, "/viver1/front/depth", 1);
    rgb_sub_.subscribe(nh_, "/viver1/front/left/color", 1);
    odom_sub_.subscribe(nh_, "/viver1/odometry", 1);

    // Approximate time synchronization for depth, RGB, and odometry
    typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry
    > MySyncPolicy;
    sync_.reset(new message_filters::Synchronizer<MySyncPolicy>(
          MySyncPolicy(10), depth_sub_, rgb_sub_, odom_sub_));
    sync_->registerCallback(boost::bind(&DepthRGBMapNode::callback, this, _1, _2, _3));

    // Publishers
    map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("voxel_map_local", 1);
    global_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("voxel_map_global", 1);
  }

private:
  // Synchronized callback for depth, RGB, and odometry messages.
  // Builds a local voxel map from the current frame, maintains a sliding window
  // of the most recent 3 frames, and publishes combined local and accumulated global maps.
  void callback(const sensor_msgs::ImageConstPtr& depth_msg,
                const sensor_msgs::ImageConstPtr& rgb_msg,
                const nav_msgs::OdometryConstPtr& odom_msg)
  {
    // Build local voxel map from the current frame
    auto local_voxel_map = processLocalMap(depth_msg, rgb_msg, odom_msg);

    // Add to sliding window (keep only the latest 3 frames)
    sliding_window_.push_back(local_voxel_map);
    if (sliding_window_.size() > 3) {
      sliding_window_.pop_front();
    }

    // Once 3 frames are collected, combine and publish
    if (sliding_window_.size() == 3) {
      auto combined_map = combineVoxelMaps(sliding_window_);
      auto local_cloud = buildCloudFromVoxelMap(combined_map);

      // Publish local voxel map
      sensor_msgs::PointCloud2 local_msg;
      pcl::toROSMsg(*local_cloud, local_msg);
      local_msg.header.stamp = depth_msg->header.stamp;
      local_msg.header.frame_id = "odom";
      map_pub_.publish(local_msg);

      ROS_INFO("Published local voxel map with %zu voxels.", combined_map.size());

      // Accumulate into global map (filter by z-threshold)
      for (const auto &kv : combined_map) {
        int vz = std::get<2>(kv.first);
        float global_z = (vz + 0.5f) * voxel_size_;
        if (global_z > max_z_threshold_) {
          continue;
        }
        global_voxel_map_[kv.first] = kv.second;
      }

      // Publish global voxel map
      auto global_cloud = buildCloudFromVoxelMap(global_voxel_map_);
      sensor_msgs::PointCloud2 global_msg;
      pcl::toROSMsg(*global_cloud, global_msg);
      global_msg.header.stamp = depth_msg->header.stamp;
      global_msg.header.frame_id = "odom";
      global_map_pub_.publish(global_msg);

      ROS_INFO("Published GLOBAL voxel map with %zu voxels.", global_voxel_map_.size());
    }
    else {
      ROS_INFO("Sliding window size = %lu. Not enough frames to build local map yet.",
                sliding_window_.size());
    }
  }

  // Processes a single frame of depth + RGB + odometry into a voxel map.
  // Back-projects depth pixels to 3D, transforms through camera-to-base and
  // base-to-global coordinate frames, and assigns RGB color to each voxel.
  std::unordered_map<std::tuple<int,int,int>, std::tuple<uint8_t,uint8_t,uint8_t>, HashTuple>
  processLocalMap(const sensor_msgs::ImageConstPtr& depth_msg,
                  const sensor_msgs::ImageConstPtr& rgb_msg,
                  const nav_msgs::OdometryConstPtr& odom_msg)
  {
    std::unordered_map<std::tuple<int,int,int>, std::tuple<uint8_t,uint8_t,uint8_t>, HashTuple> voxel_map;

    // Convert depth and RGB images via cv_bridge
    cv_bridge::CvImageConstPtr cv_depth_ptr, cv_rgb_ptr;
    try {
      cv_depth_ptr = cv_bridge::toCvShare(depth_msg);
    } catch(const cv_bridge::Exception &e) {
      ROS_WARN("Depth exception: %s", e.what());
      return voxel_map;
    }
    try {
      cv_rgb_ptr = cv_bridge::toCvShare(rgb_msg, "bgr8");
    } catch(const cv_bridge::Exception &e) {
      ROS_WARN("RGB exception: %s", e.what());
      return voxel_map;
    }

    // Apply depth edge filter to remove noisy depth discontinuities
    cv::Mat depth_filtered = cv_depth_ptr->image.clone();
    if (depth_filtered.type() == CV_16UC1) {
      filterDepthEdge(depth_filtered, max_range_);
    }

    int width  = depth_filtered.cols;
    int height = depth_filtered.rows;

    // Build odometry-based global transform (base to global)
    double tx = odom_msg->pose.pose.position.x;
    double ty = odom_msg->pose.pose.position.y;
    double tz = odom_msg->pose.pose.position.z;
    double qx = odom_msg->pose.pose.orientation.x;
    double qy = odom_msg->pose.pose.orientation.y;
    double qz = odom_msg->pose.pose.orientation.z;
    double qw = odom_msg->pose.pose.orientation.w;

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    {
      Eigen::Quaternionf quat((float)qw, (float)qx, (float)qy, (float)qz);
      Eigen::Matrix3f rot3 = quat.toRotationMatrix();
      transform.block<3,3>(0,0) = rot3;
      transform(0,3) = (float)tx;
      transform(1,3) = (float)ty;
      transform(2,3) = (float)tz;
    }

    // Camera-to-base extrinsic rotation: Z->X, X->-Y, Y->-Z
    Eigen::Matrix4f T_cam_to_base = Eigen::Matrix4f::Identity();
    if (apply_cam_to_base_) {
      Eigen::Matrix3f R;
      R <<  0,  0, 1,
           -1,  0, 0,
            0, -1, 0;
      T_cam_to_base.block<3,3>(0,0) = R;
    }

    // Back-project each pixel to 3D and assign to a voxel
    for(int v = 0; v < height; v++) {
      for(int u = 0; u < width; u++) {
        float raw_depth = 0.0f;

        if(depth_filtered.type() == CV_16UC1) {
          uint16_t d_raw = depth_filtered.at<uint16_t>(v, u);
          // Skip pixels invalidated by edge filter
          if(d_raw == std::numeric_limits<uint16_t>::max()) {
            continue;
          }
          raw_depth = static_cast<float>(d_raw) / factor_;
        }
        else if(depth_filtered.type() == CV_32FC1) {
          float d_raw = depth_filtered.at<float>(v, u);
          raw_depth = d_raw / factor_;
        }

        if(raw_depth < 0.0001f) {
          continue;
        }

        float Z = raw_depth;
        float X = (u - cx_) * Z / fx_;
        float Y = (v - cy_) * Z / fy_;

        Eigen::Vector4f pt_cam(X, Y, Z, 1.0f);

        // Range filtering
        float dist = pt_cam.head<3>().norm();
        if(dist > max_range_) {
          continue;
        }

        // Extract RGB color
        cv::Vec3b bgr = cv_rgb_ptr->image.at<cv::Vec3b>(v,u);
        uint8_t b = bgr[0];
        uint8_t g = bgr[1];
        uint8_t r = bgr[2];

        // Transform: camera -> base -> global
        Eigen::Vector4f pt_base = apply_cam_to_base_ ? (T_cam_to_base * pt_cam) : pt_cam;
        Eigen::Vector4f pt_global = transform * pt_base;

        // Compute voxel coordinates
        int vx = static_cast<int>(std::floor(pt_global.x() / voxel_size_));
        int vy = static_cast<int>(std::floor(pt_global.y() / voxel_size_));
        int vz = static_cast<int>(std::floor(pt_global.z() / voxel_size_));

        std::tuple<int,int,int> key(vx, vy, vz);
        // Overwrite with the latest observed color
        voxel_map[key] = std::make_tuple(r, g, b);
      }
    }
    return voxel_map;
  }

  // Merges multiple voxel maps from the sliding window into a single combined map.
  // Later observations overwrite earlier ones for the same voxel key.
  std::unordered_map<std::tuple<int,int,int>, std::tuple<uint8_t,uint8_t,uint8_t>, HashTuple>
  combineVoxelMaps(const std::deque<
    std::unordered_map<std::tuple<int,int,int>, std::tuple<uint8_t,uint8_t,uint8_t>, HashTuple>
  > &maps)
  {
    std::unordered_map<std::tuple<int,int,int>, std::tuple<uint8_t,uint8_t,uint8_t>, HashTuple> combined;
    for (auto &m : maps) {
      for (auto &kv : m) {
        combined[kv.first] = kv.second;
      }
    }
    return combined;
  }

  // Converts a voxel map into a PCL XYZRGB point cloud for ROS publishing.
  // Each voxel center is placed at (vx + 0.5) * voxel_size.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  buildCloudFromVoxelMap(const std::unordered_map<
    std::tuple<int,int,int>, std::tuple<uint8_t,uint8_t,uint8_t>, HashTuple> &voxel_map)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->points.reserve(voxel_map.size());
    for(const auto &kv : voxel_map) {
      auto vox = kv.first;
      auto rgb = kv.second;
      int vx = std::get<0>(vox);
      int vy = std::get<1>(vox);
      int vz = std::get<2>(vox);

      float cx_voxel = (vx + 0.5f) * voxel_size_;
      float cy_voxel = (vy + 0.5f) * voxel_size_;
      float cz_voxel = (vz + 0.5f) * voxel_size_;

      pcl::PointXYZRGB p;
      p.x = cx_voxel;
      p.y = cy_voxel;
      p.z = cz_voxel;
      p.r = std::get<0>(rgb);
      p.g = std::get<1>(rgb);
      p.b = std::get<2>(rgb);
      cloud->points.push_back(p);
    }
    cloud->width  = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
  }

private:
  ros::NodeHandle nh_;

  // Camera intrinsic parameters
  float fx_, fy_, cx_, cy_;
  float factor_;
  float voxel_size_;
  float max_range_;
  bool  apply_cam_to_base_;
  float max_z_threshold_;

  // Sliding window storing the most recent 3 frame voxel maps
  std::deque<
    std::unordered_map<std::tuple<int,int,int>, std::tuple<uint8_t,uint8_t,uint8_t>, HashTuple>
  > sliding_window_;

  // Accumulated global voxel map
  std::unordered_map<
    std::tuple<int,int,int>, std::tuple<uint8_t,uint8_t,uint8_t>, HashTuple
  > global_voxel_map_;

  // Synchronized subscribers
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
  message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
  typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry
  > SyncPolicy;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  // Publishers
  ros::Publisher map_pub_;
  ros::Publisher global_map_pub_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "depth_rgb_map_node");
  ros::NodeHandle nh("~");
  DepthRGBMapNode node(nh);
  ros::spin();
  return 0;
}
