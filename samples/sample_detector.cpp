#include "class_timer.hpp"
#include "class_detector.h"

#include <memory>
#include <thread>

std::unique_ptr<Detector>test1(new Detector());
int main()
{
		
	Config config_v5;
	config_v5.net_type = YOLOV5;
	config_v5.detect_thresh = 0.5;
	config_v5.file_model_cfg = "../configs/yolov5-6.0/yolov5s.cfg";
	config_v5.file_model_weights = "../configs/yolov5-6.0/yolov5s.weights";
	config_v5.calibration_image_list_file_txt = "../configs/calibration_images.txt";
	config_v5.inference_precison = FP32;


	test1->init(config_v5);
	cv::Mat image0 = cv::imread("../configs/dog.jpg", cv::IMREAD_UNCHANGED);
	cv::Mat image1 = cv::imread("../configs/person.jpg", cv::IMREAD_UNCHANGED);
	std::vector<BatchResult> batch_res;
	Timer timer;
	for (;;)
	{
		//prepare batch data
		std::vector<cv::Mat> batch_img;
		cv::Mat temp0 = image0.clone();
		cv::Mat temp1 = image1.clone();
		batch_img.push_back(temp0);
		//batch_img.push_back(temp1);

		//detect
		timer.reset();
		test1->detect(batch_img, batch_res);
		timer.out("detect");

		//disp
		for (int i=0;i<batch_img.size();++i)
		{
			for (const auto &r : batch_res[i])
			{
				std::cout <<"batch "<<i<< " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
				cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);
				std::stringstream stream;
				stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
				cv::putText(batch_img[i], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
			}
			cv::namedWindow("image" + std::to_string(i), cv::WINDOW_NORMAL);
			cv::imshow("image"+std::to_string(i), batch_img[i]);
		}
		cv::waitKey(10);
	}
}
