#include "class_timer.hpp"
#include "class_detector.h"

#include <memory>
#include <thread>

#include <memory>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>


#include <ros/ros.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <camera_info_manager/camera_info_manager.h>
#include <image_transport/image_transport.h>


#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>

#include <bounding_box_msgs/boundingbox.h>
#include <bounding_box_msgs/boundingboxes.h>


#include <iostream>

#include <fstream>
#include "yaml-cpp/yaml.h"

class ObjectDetector
{
public:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::CameraSubscriber img_sub_;
    image_transport::CameraSubscriber depth_sub_;
    image_transport::Publisher pub_;

    ros::Publisher boundingbx_pub_;
    ros::Publisher boundingbxs_pub_;


    std::vector<std::string> frame_ids_;
    std::unique_ptr<Detector> objdetector;
    std::string const package_path;
    
    std::string sub_img_topic_name;
    std::string pub_bbximg_topic_name;
    std::string classes_yaml_path;
    double conv_tsh_val;
    int y_detection_border;
    
    std::vector<std::string> names;
    

    ObjectDetector(): it_(nh_),objdetector (std::make_unique<Detector>())
    {
        std::string const package_path = ros::package::getPath("object_detection");
        
        ros::param::get("/conv_tsh", conv_tsh_val);
        ros::param::get("/TopicNamePubImgRaw", pub_bbximg_topic_name);
        ros::param::get("/TopicNameSubImgRaw", sub_img_topic_name);
        ros::param::get("/y_detect_border", y_detection_border);
        ros::param::get("/classnameyamlpath", classes_yaml_path);
        
        
        
        classes_yaml_path = package_path + classes_yaml_path;

        std::string image_topic = nh_.resolveName("/front_camera/color/image_raw");
        std::string image_depth_topic = nh_.resolveName("/front_camera/color/image_depth");
        
        img_sub_ = it_.subscribeCamera(sub_img_topic_name, 1, &ObjectDetector::imageCallback, this);
        //sub_ = it_.subscribeCamera(image_depth_topic, 1, &ObjectDetector::imagedepthCallback, this);
        
        pub_ = it_.advertise(pub_bbximg_topic_name, 10);
        
        boundingbx_pub_ = nh_.advertise<bounding_box_msgs::boundingbox>("object_detector/detections/boundingbox", 10);
        boundingbxs_pub_ = nh_.advertise<bounding_box_msgs::boundingboxes>("object_detector/detections/boundingboxs", 10);
        
        YAML::Node config = YAML::LoadFile(classes_yaml_path);
        

	if (config["names"]) {
  	    names = config["names"].as<std::vector<std::string>>();
        yolov5Init();
        ROS_INFO("[object_detection] Detector is ready");
    }
    else{

        ROS_ERROR("[object_detection] No Klass Names in yaml-file!");
    
    }
    }

    
    
    
    
    
    void yolov5Init()
    {
        std::string const package_path = ros::package::getPath("object_detection");
        Config config_v5;
        config_v5.net_type = YOLOV5;
        config_v5.detect_thresh = conv_tsh_val;
        config_v5.file_model_cfg = package_path+"/configs/yolov5-6.0/yolov5l.cfg";
        config_v5.file_model_weights = package_path+"/configs/yolov5-6.0/fre_22_train_01_25_05_22.weights";
        config_v5.calibration_image_list_file_txt = package_path+"/configs/calibration_images.txt";
        config_v5.inference_precison = FP32;
        
        objdetector->init(config_v5);
        
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info_msg)
    {
        
        cv::Mat image;
        cv_bridge::CvImagePtr input_bridge;
        bounding_box_msgs::boundingbox bbx;

        try {
            input_bridge = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);

            image = input_bridge->image;
            

        }
        catch (cv_bridge::Exception& ex){
            ROS_ERROR("[object_detection] Failed to convert image");
            return;
        }
        
        
        //prepare batch data
        std::vector<cv::Mat> batch_img;
	
        std::vector<BatchResult> batch_res;
        
        cv::Size s = image.size();
        cv::Rect detectionROI(0, y_detection_border, s.width-0, s.width - y_detection_border);
        cv::rectangle(image, detectionROI, cv::Scalar(0, 255, 0), 4);
        cv::Mat temp0 = image.clone();
        
        batch_img.push_back(temp0);
        
        //do detection process
        objdetector->detect(batch_img, batch_res);
        
        //draw Boundingboxes which are in the detecion area
	    for (const auto &r : batch_res[0])
	    {
	    	cv::Point center_of_rect = (r.rect.br() + r.rect.tl())*0.5;

		    if(center_of_rect.y >= y_detection_border)
		    {
            bbx.Class=r.id;
            bbx.probability=r.prob;
            bbx.rect_xmin=r.rect.x ;
            bbx.rect_ymin=r.rect.y ;
            bbx.rect_height=r.rect.height;
            bbx.rect_width=r.rect.width;
            
            std::cout << " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
            
            cv::rectangle(image, r.rect, cv::Scalar(255, 0, 0), 2);
            
            std::stringstream stream;
            
            //stream << std::fixed << std::setprecision(2)  << "  score:" << r.prob <<  " "<< names[r.id];
            
            cv::putText(image, stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 1, cv::Scalar(0, 0, 255), 2);
            
            boundingbx_pub_.publish(bbx);
            }
        }

        pub_.publish(input_bridge->toImageMsg());

        
    }

    void imagedepthCallback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info_msg)
    {
        
        cv::Mat image;
        cv_bridge::CvImagePtr input_bridge;

        bounding_box_msgs::boundingbox bbx;

        try {
            input_bridge = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
            image = input_bridge->image;
            

        }
        catch (cv_bridge::Exception& ex){
            ROS_ERROR("[object_detection] Failed to convert image");
            return;
        }
        
        //cam_model_.fromCameraInfo(info_msg);

        //prepare batch data
        std::vector<cv::Mat> batch_img;
        std::vector<BatchResult> batch_res;
        cv::Mat temp0 = image.clone();
        batch_img.push_back(temp0);
        //Timer timer;
        //do detection process
        //timer.reset();
        objdetector->detect(batch_img, batch_res);
        //timer.out("detect");

        //disp
        for (int i=0;i<batch_img.size();++i)
        {            
            for (const auto &r : batch_res[i])
            {
                
                bbx.Class=r.id;
                bbx.probability=r.prob;
                bbx.rect_xmin=r.rect.x ;
                bbx.rect_ymin=r.rect.y ;
                bbx.rect_height=r.rect.height;
                bbx.rect_width=r.rect.width;
                std::cout <<"batch "<<i<< " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
                cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);
                std::stringstream stream;
                stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
                cv::putText(batch_img[i], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
                boundingbx_pub_.publish(bbx);
            }

        pub_.publish(input_bridge->toImageMsg());
        }
        
    }
};



int main(int argc, char** argv)
{
  ros::init(argc, argv, "object_detector");
  std::vector<std::string> frame_ids(argv + 1, argv + argc);
  ROS_INFO("[object_detection] Detector started, load model...");
  ObjectDetector detector;
  ros::spin();
  
}
