#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <numeric>
#include <vector>

using namespace cv;
using namespace std;

struct GaussianDistribution
{
	float w;
	float u;
	float sigma;
};

struct PixelGMM
{
    int num;// 高斯模型最大数量
    int index;// 当前使用的高斯模型数量
    GaussianDistribution* gd;// 高斯模型数组指针
};
bool modelInit = false;
const int GAUSSIAN_MODULE_NUMS = 5;// 模型数量
const float ALPHA = 0.005;// 学习速率
const float SIGMA = 30;// 标准差初始值
const float WEIGHT = 0.05;// 高斯模型权重初始值
const float T = 0.7;// 有效高斯分布阈值
int rows, cols;
PixelGMM* ppgmm;

int main(int argc, char *argv[])
{
	VideoCapture capture("video.avi");
	Ptr<BackgroundSubtractorMOG2> bg_model = createBackgroundSubtractorMOG2(100,SIGMA,false);
	Mat rate, image, fgimage, fgmask;
	bg_model->setVarThreshold(20);
	cv::Size s;
	s.width = 860; 
	s.height = 540;

	while(1)
	{
		int niters = 1;
            	vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		
		capture >> rate;
		if(!rate.data)
		{
			cerr << "picture error!";
			return -1;
		}
		resize(rate, image, s, INTER_CUBIC);

		if (fgimage.empty())
                fgimage.create(image.size(), image.type());

            	bg_model->apply(image, fgmask, ALPHA);
            	fgimage = Scalar::all(0);
            	image.copyTo(fgimage, fgmask);
            	Mat bgimage;
            	bg_model->getBackgroundImage(bgimage);
		
		Mat temp;
		
		dilate(fgmask, temp, Mat(), Point(-1,-1), niters);//膨胀，3*3的element，迭代次数为niters
            	erode(temp, temp, Mat(), Point(-1,-1), niters*2);//腐蚀
            	dilate(temp, temp, Mat(), Point(-1,-1), niters);
		
		findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );//找轮廓
            	fgimage = Mat::zeros(image.size(), CV_8UC3);
		
		if(contours.size() > 0)
		{
			float Area;
			Rect rect;
			vector<Point> m;
			for(int i = contours.size() - 1; i >= 0; i--)
			{
				vector<Point> c = contours[i];
                    		//获取面积
                    		Area = contourArea(c);
                   	 	if (Area < 10)//50这个值根据需求设定，这里指的是目标的大小
                    		{
                        		continue;
                    		}
                    		else
                    		{
                        		m = c;
                    		}
                    		rect = boundingRect(m);
                    		rectangle(image, rect, Scalar(0, 255, 0), 2);
			}
			Scalar color(0, 255, 0);
			drawContours(fgimage, contours, -1, color, CV_FILLED, 8, hierarchy);
		}

		imshow("image", image);
            	imshow("fgimage", fgimage);
            	imshow("fgmask", fgmask);
            	if (!bgimage.empty())
               		imshow("bgimage", bgimage);

            	waitKey(30);
	}
	return 0;
}

