#include "class_timer.hpp"
#include "class_detector.h"

#include <memory>
#include <thread>

#include <memory>
#include <string>
#include <map>


#include<math.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <algorithm>
#include <vector>
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
    int tracking_closest_tsh_pixel;
    int MaxObjAge;
    int y_detection_border;

    struct CenterPoint{
        float CpX;
        float CpY;
    };

    struct DistId{
        double Dist;
        int ObjIdFrom;
        int ObjIdTo;
    };


    struct Object{
        int DictID;
        int ClassSpecID;
        int Age;
        std::string ClassName;
        float Prob;
        int RectXmin;
        int RectYmin;
        int RectHeight;
        int RectWidth;
        CenterPoint Center;
    };

    typedef std::pair<int, DistId> DistDictType;


    int ObjCnt = 0;

    std::map<int, Object> ObjectDict;
    
    std::vector<std::string> names;
    

    ObjectDetector(): it_(nh_),objdetector (std::make_unique<Detector>())
    {
        std::string const package_path = ros::package::getPath("object_detection");
        
        ros::param::get("/conv_tsh", conv_tsh_val);
        ros::param::get("/tracking_closest_tsh_pixel", tracking_closest_tsh_pixel);
        ros::param::get("/tracking_max_obj_age", MaxObjAge);
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
        bounding_box_msgs::boundingboxes bbxs;
        std_msgs::Header h = image_msg->header;
        bbxs.header = h;
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

        std::map<int, Object> TmpObjDict;

        for (const auto &r : batch_res[0])
        {
            cv::Point center_of_rect = (r.rect.br() + r.rect.tl())*0.5;

            if(center_of_rect.y >= y_detection_border)
            {

                Object Tmp;
                Tmp.ClassName = names[r.id];
                Tmp.DictID = ObjCnt;
                Tmp.Age = 0;
                Tmp.Prob=r.prob;
                Tmp.RectXmin=r.rect.x ;
                Tmp.RectYmin=r.rect.y ;
                Tmp.RectHeight=r.rect.height;
                Tmp.RectWidth=r.rect.width;
                Tmp.Center.CpX = r.rect.x + (r.rect.width/2);
                Tmp.Center.CpY = r.rect.y + (r.rect.height/2);

                TmpObjDict.insert ( std::pair<int,Object>(ObjCnt,Tmp) );

                ObjCnt++;
            }

        }

        if(!ObjectDict.empty())
        {
            std::cout << "Current Emelents " << ObjectDict.size() << '\n';
            std::vector<int> MachedIdsTo;
            std::vector<int> MachedIdsFrom;
            if (!TmpObjDict.empty()){

                for (auto const& l : TmpObjDict){
                    
                    std::cout << "Temp Detected Emelents " << TmpObjDict.size() << '\n';
                    Object TmpObj = l.second;
                    int tmpid = l.first;

                    std::map<int, DistId> DistDict;     //In DistDict the ID (key) is the Same ID as in the TmpObjDict

                    for (auto const& x : ObjectDict)    //Get all distances of the same class to each other
                    {   
                        int Id = x.first;
                        Object Obj = x.second;
                        
                        if(Obj.ClassName == TmpObj.ClassName) //Object is same Class
                        {
                            DistDictType DistElm;
                            DistElm.first = tmpid;
                            DistElm.second.ObjIdFrom = tmpid;
                            DistElm.second.ObjIdTo = Id;
                            DistElm.second.Dist = EuklideanDist(TmpObj,Obj);
                            
                            DistDict.insert (std::pair<int,DistId>(tmpid,DistElm.second));


                        }
                    }

                    if(!DistDict.empty()){

                        DistDictType clst = GetClstId(DistDict);

                        DistId  temp = clst.second;
                        
                        if ((int)clst.second.Dist < tracking_closest_tsh_pixel){

                            ObjectDict[clst.second.ObjIdTo] = TmpObjDict[clst.second.ObjIdFrom];
                            Object MatchObj = ObjectDict[clst.second.ObjIdTo];

                            MachedIdsFrom.push_back(clst.second.ObjIdFrom);

                            MachedIdsTo.push_back(clst.second.ObjIdTo);

                            if (DistDict.count(clst.second.ObjIdFrom) > 0) {
                                
                                DistDict.erase(clst.second.ObjIdFrom);

                            }

                        }
                    }
                    else{       //IF there is nothing in the current object which is the same class, add it!

                        ObjectDict.insert(std::pair<int,Object>(ObjCnt,TmpObj) );


                    }
                }

                for(auto it = std::begin(MachedIdsFrom); it != std::end(MachedIdsFrom); ++it) {
                    if (TmpObjDict.count(*it) > 0) {
                        TmpObjDict.erase(*it);
                    }
                }

            
            std::cout << "Aftermatching Temp Elements " << TmpObjDict.size() << '\n';
            std::cout << "Aftermatching Object Elements " << ObjectDict.size() << '\n';

            }
            
            if(!TmpObjDict.empty()){
                for (auto const& x : TmpObjDict){

                    if(x.second.Center.CpY >1000){ //It seems the object has left the Window in y-direction
                        if (TmpObjDict.count(x.first) > 0) {
                        TmpObjDict.erase(x.first);
                        }
                        

                    }
                    else{

                        ObjectDict.insert ( std::pair<int,Object>(x.first,x.second) );

                    }

                    

                }
            }
            //Get all unmatched objects in the dict:
            std::vector<int> UnmatchedIds;
            std::vector<int> ToRemoveObjs;
            for(auto const& x : ObjectDict){

                int item = x.first;

                if ( std::find(MachedIdsTo.begin(), MachedIdsTo.end(), item) != MachedIdsTo.end() ){
                    //Do that                
                }
                else{
                    if (x.second.Age >= MaxObjAge)
                    {
                        ToRemoveObjs.push_back(item);
                    }
                    else{
                        Object temp = x.second;
                        temp.Age++;
                        ObjectDict[item] = temp;
                    } 
                }
  
            }

            for(auto it = std::begin(ToRemoveObjs); it != std::end(ToRemoveObjs); ++it) {

                if (ObjectDict.count(*it) > 0) {
                    ObjectDict.erase(*it);
                }
            }

            


            for(auto it = std::begin(MachedIdsTo); it != std::end(MachedIdsTo); ++it) {

                if (TmpObjDict.count(*it) > 0) {

                    TmpObjDict.erase(*it);
                }
            }

        }

        else{

            if(!TmpObjDict.empty()){

                for (auto const& x : TmpObjDict){

                    ObjectDict.insert ( std::pair<int,Object>(x.first,x.second) );

                }

            }

        }

        
        for (auto const& h : ObjectDict){

            Object Obj = h.second;
            int id = h.first;
                
            bbx.Class=Obj.ClassName;
            bbx.probability=Obj.Prob;
            bbx.rect_xmin=Obj.RectXmin;
            bbx.rect_ymin=Obj.RectYmin;
            bbx.rect_height=Obj.RectHeight;
            bbx.rect_width=Obj.RectWidth;
            
            cv::Rect rect1;
            rect1.x = Obj.RectXmin;
            rect1.y = Obj.RectYmin;
            rect1.width = Obj.RectWidth;
            rect1.height = Obj.RectHeight;
            
            
            cv::rectangle(image, rect1, cv::Scalar(255, 0, 0), 2);
            
            std::stringstream stream;
            
            stream << std::fixed << std::setprecision(2)  << "  score:" << Obj.Prob <<  " "<< Obj.ClassName << "id:" << id;
            
            cv::putText(image, stream.str(), cv::Point(Obj.RectXmin, Obj.RectYmin - 5), 0, 1, cv::Scalar(0, 0, 255), 2);

            boundingbx_pub_.publish(bbx);
            bbxs.bounding_boxes.push_back(bbx);
            }

            boundingbxs_pub_.publish(bbxs);
            pub_.publish(input_bridge->toImageMsg());

            
    }

    double EuklideanDist(Object Obj1, Object Obj2)
    {

        double dist = sqrt(pow((Obj1.Center.CpX - Obj2.Center.CpX), 2) + pow((Obj1.Center.CpY - Obj2.Center.CpY), 2));      

        return dist;
    }




    struct CompareSecond
    {
        bool operator()(const DistDictType& left, const DistDictType& right) const
        {
            return left.second.Dist < right.second.Dist;
        }
    };

    DistDictType GetClstId(std::map<int, DistId> DistDict) 
    {
    std::pair<int,DistId > min = *min_element(DistDict.begin(), DistDict.end(), CompareSecond());
    return min; 
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
