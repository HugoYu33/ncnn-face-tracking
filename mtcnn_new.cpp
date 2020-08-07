#include "net.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <sys/time.h>

//#define SHOW_FPS 1

using namespace std;
using namespace cv;

static int NEW_DATA = 1; // using sunplus optimized model data
static int P_POINTS = 0; // process points

struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    float ppoint[10];
    float regreCoord[4];
};

struct orderScore
{
    float score;
    int oriOrder;
};
bool cmpScore(orderScore lsh, orderScore rsh){
    if(lsh.score<rsh.score)
        return true;
    else
        return false;
}
#ifdef SHOW_FPS
static float getElapse(struct timeval *tv1,struct timeval *tv2)
{
	uint r = (tv2->tv_usec - tv1->tv_usec) + (tv2->tv_sec - tv1->tv_sec) * 1000000;
	//printf("%d\n", r);
    return (float)r / 1000000.0f;
}
#endif
class mtcnn{
public:
    mtcnn();
    void detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox);
private:
    void generateBbox(ncnn::Mat score, ncnn::Mat location, vector<Bbox>& boundingBox_, vector<orderScore>& bboxScore_, float scale);
    void nms(vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname="Union");
    void refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width);

    ncnn::Net Pnet, Rnet, Onet;
    ncnn::Mat img;

    const float nms_threshold[3] = {0.5, 0.7, 0.7};
    const float threshold[3] = {0.8, 0.8, 0.6};
    const float mean_vals[3] = {127.5, 127.5, 127.5};
    const float norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};
    std::vector<Bbox> firstBbox_, secondBbox_,thirdBbox_;
    std::vector<orderScore> firstOrderScore_, secondBboxScore_, thirdBboxScore_;
    int img_w, img_h;
};

mtcnn::mtcnn(){
	if (NEW_DATA) {
		Pnet.load_param("./models/PNet_new2.param");
		Pnet.load_model("./models/PNet_new2.bin");
		Rnet.load_param("./models/RNet_new2.param");
		Rnet.load_model("./models/RNet_new2.bin");
		Onet.load_param("./models/ONet_new2.param");
		Onet.load_model("./models/ONet_new2.bin");
	} else {
		Pnet.load_param("./models/det1.param");
		Pnet.load_model("./models/det1.bin");
		Rnet.load_param("./models/det2.param");
		Rnet.load_model("./models/det2.bin");
		Onet.load_param("./models/det3.param");
		Onet.load_model("./models/det3.bin");
	}
}

void mtcnn::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, std::vector<orderScore>& bboxScore_, float scale){
    int stride = 2;
    int cellsize = 12;
    int count = 0;
    //score p
    float *p = score.channel(1);
    float *plocal = location.channel(0);
    Bbox bbox;
    float inv_scale = 1.0f/scale;
    orderScore order;
    for(int row=0;row<score.h;row++){
        for(int col=0;col<score.w;col++){
            if(*p>threshold[0]){
                bbox.score = *p;
                order.score = *p;
                order.oriOrder = count;
                bbox.x1 = round((stride*col+1)*inv_scale);
                bbox.y1 = round((stride*row+1)*inv_scale);
                bbox.x2 = round((stride*col+1+cellsize)*inv_scale);
                bbox.y2 = round((stride*row+1+cellsize)*inv_scale);
                bbox.exist = true;
                bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
                for(int channel=0;channel<4;channel++)
                    bbox.regreCoord[channel]=location.channel(channel)[0];
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;
            }
            p++;
            plocal++;
        }
    }
}
void mtcnn::nms(std::vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname){
    if(boundingBox_.empty()){
        return;
    }
    std::vector<int> heros;
    //sort the score
    sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);

    int order = 0;
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    //std::multimap<float, int> vScores;
    while(bboxScore_.size()>0){
        order = bboxScore_.back().oriOrder;
        bboxScore_.pop_back();
        if(order<0)continue;
        if(boundingBox_.at(order).exist == false) continue;
        heros.push_back(order);
        boundingBox_.at(order).exist = false;//delete it

        for(int num=0;num<boundingBox_.size();num++){
            if(boundingBox_.at(num).exist){
                //the iou
                maxX = (boundingBox_.at(num).x1>boundingBox_.at(order).x1)?boundingBox_.at(num).x1:boundingBox_.at(order).x1;
                maxY = (boundingBox_.at(num).y1>boundingBox_.at(order).y1)?boundingBox_.at(num).y1:boundingBox_.at(order).y1;
                minX = (boundingBox_.at(num).x2<boundingBox_.at(order).x2)?boundingBox_.at(num).x2:boundingBox_.at(order).x2;
                minY = (boundingBox_.at(num).y2<boundingBox_.at(order).y2)?boundingBox_.at(num).y2:boundingBox_.at(order).y2;
                //maxX1 and maxY1 reuse 
                maxX = ((minX-maxX+1)>0)?(minX-maxX+1):0;
                maxY = ((minY-maxY+1)>0)?(minY-maxY+1):0;
                //IOU reuse for the area of two bbox
                IOU = maxX * maxY;
                if(!modelname.compare("Union"))
                    IOU = IOU/(boundingBox_.at(num).area + boundingBox_.at(order).area - IOU);
                else if(!modelname.compare("Min")){
                    IOU = IOU/((boundingBox_.at(num).area<boundingBox_.at(order).area)?boundingBox_.at(num).area:boundingBox_.at(order).area);
                }
                if(IOU>overlap_threshold){
                    boundingBox_.at(num).exist=false;
                    for(vector<orderScore>::iterator it=bboxScore_.begin(); it!=bboxScore_.end();it++){
                        if((*it).oriOrder == num) {
                            (*it).oriOrder = -1;
                            break;
                        }
                    }
                }
            }
        }
    }
    for(int i=0;i<heros.size();i++)
        boundingBox_.at(heros.at(i)).exist = true;
}
void mtcnn::refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width){
    if(vecBbox.empty()){
        cout<<"Bbox is empty!!"<<endl;
        return;
    }
    float bbw=0, bbh=0, maxSide=0;
    float h = 0, w = 0;
    float x1=0, y1=0, x2=0, y2=0;
    for(vector<Bbox>::iterator it=vecBbox.begin(); it!=vecBbox.end();it++){
        if((*it).exist){
            bbw = (*it).x2 - (*it).x1 + 1;
            bbh = (*it).y2 - (*it).y1 + 1;
            x1 = (*it).x1 + (*it).regreCoord[0]*bbw;
            y1 = (*it).y1 + (*it).regreCoord[1]*bbh;
            x2 = (*it).x2 + (*it).regreCoord[2]*bbw;
            y2 = (*it).y2 + (*it).regreCoord[3]*bbh;

            w = x2 - x1 + 1;
            h = y2 - y1 + 1;

            maxSide = (h>w)?h:w;
            x1 = x1 + w*0.5 - maxSide*0.5;
            y1 = y1 + h*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);

            //boundary check
            if((*it).x1<0)(*it).x1=0;
            if((*it).y1<0)(*it).y1=0;
            if((*it).x2>width)(*it).x2 = width - 1;
            if((*it).y2>height)(*it).y2 = height - 1;

            it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
        }
    }
}
void mtcnn::detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox_){
    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();

    img = img_;
    img_w = img.w;
    img_h = img.h;
    img.substract_mean_normalize(mean_vals, norm_vals);

    float minl = img_w<img_h?img_w:img_h;
    int MIN_DET_SIZE = 12;
    int minsize = 90;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = 0.709;
    int factor_count = 0;
    vector<float> scales_;
    while(minl>MIN_DET_SIZE){
        if(factor_count>0)m = m*factor;
        scales_.push_back(m);
        minl *= factor;
        factor_count++;
    }
    orderScore order;
    int count = 0;

    for (size_t i = 0; i < scales_.size(); i++) {
        int hs = (int)ceil(img_h*scales_[i]);
        int ws = (int)ceil(img_w*scales_[i]);
        //ncnn::Mat in = ncnn::Mat::from_pixels_resize(image_data, ncnn::Mat::PIXEL_RGB2BGR, img_w, img_h, ws, hs);
        ncnn::Mat in;
        resize_bilinear(img_, in, ws, hs);
        //in.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = Pnet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score_, location_;
        ex.extract("prob1", score_);
        ex.extract("conv4-2", location_);
        std::vector<Bbox> boundingBox_;
        std::vector<orderScore> bboxScore_;
        generateBbox(score_, location_, boundingBox_, bboxScore_, scales_[i]);
        nms(boundingBox_, bboxScore_, nms_threshold[0]);

        for(vector<Bbox>::iterator it=boundingBox_.begin(); it!=boundingBox_.end();it++){
            if((*it).exist){
                firstBbox_.push_back(*it);
                order.score = (*it).score;
                order.oriOrder = count;
                firstOrderScore_.push_back(order);
                count++;
            }
        }
        bboxScore_.clear();
        boundingBox_.clear();
    }
    //the first stage's nms
    if(count<1)return;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
    refineAndSquareBbox(firstBbox_, img_h, img_w);
    //printf("firstBbox_.size()=%d\n", firstBbox_.size());

    //second stage
    count = 0;
    for(vector<Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        if((*it).exist){
            ncnn::Mat tempIm;
            copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
            ncnn::Mat in;
            resize_bilinear(tempIm, in, 24, 24);
            ncnn::Extractor ex = Rnet.create_extractor();
            ex.set_light_mode(true);
            ex.input("data", in);
            ncnn::Mat score, bbox;
            ex.extract("prob1", score);
            ex.extract(NEW_DATA?"fc5-2":"conv5-2", bbox);
            if((score[1])>threshold[1]){
                for(int channel=0;channel<4;channel++)
                    it->regreCoord[channel]=bbox[channel];
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = score[1];
                secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                secondBboxScore_.push_back(order);
            }
            else{
                (*it).exist=false;
            }
        }
    }
    //printf("secondBbox_.size()=%d\n", secondBbox_.size());
    if(count<1)return;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
    refineAndSquareBbox(secondBbox_, img_h, img_w);

    //third stage 
    count = 0;
    for(vector<Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            ncnn::Mat tempIm;
            copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
            ncnn::Mat in;
            resize_bilinear(tempIm, in, 48, 48);
            ncnn::Extractor ex = Onet.create_extractor();
            ex.set_light_mode(true);
            ex.input("data", in);
            ncnn::Mat score, bbox, keyPoint;
            ex.extract("prob1", score);
            ex.extract(NEW_DATA?"fc6-2":"conv6-2", bbox);
            ex.extract(NEW_DATA?"fc6-3":"conv6-3", keyPoint);
            if(score[1]>threshold[2]){
                for(int channel=0;channel<4;channel++)
                    it->regreCoord[channel]=bbox[channel];
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = score[1];
                for(int num=0;num<5;num++){
                    (it->ppoint)[num] = it->x1 + (it->x2 - it->x1)*keyPoint[num];
                    (it->ppoint)[num+5] = it->y1 + (it->y2 - it->y1)*keyPoint[num+5];
                }

                thirdBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                thirdBboxScore_.push_back(order);
            }
            else
                (*it).exist=false;
            }
        }

    //printf("thirdBbox_.size()=%d\n", thirdBbox_.size());
    if(count<1)return;
    refineAndSquareBbox(thirdBbox_, img_h, img_w);
    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
    finalBbox_ = thirdBbox_;
}

#define PWM_CYCLE   (3030000)//=1000000000/3030000 ~= 330Hz (period ~=3030 us)
//3 0.05 0.5
#define KP_X  3     //X axis motor P control gain     
#define KI_X  0.1   //X axis motor I control gain    
#define KD_X  0.5   //X axis motor D control gain    
#define KP_Y  2     //Y axis motor P control gain    
#define KI_Y  0.1   //Y axis motor I control gain    
#define KD_Y  0.5   //Y axis motor D control gain    
#define DEAD_ZONE_X  0.1   //X axis control target torlence
#define DEAD_ZONE_Y  0.1   //Y axis control target torlence
#define MOTOR_CENTER_PERIOD      1500000  //duty dac value=1500000, physical meaning = 1500 us, with this control pulse motor is in  90 degree position
#define MOTOR_UP_LIMIT_PERIOD    2500000  //duty dac value=2500000, physical meaning = 2500 us, with this control pulse motor is in 180 degree position
#define MOTOR_DOWN_LIMIT_PERIOD   500000  //duty dac value= 500000, physical meaning =  500 us, with this control pulse motor is in   0 degree position
#define PWM_DAC_REVOLUTION = PWM_CYCLE/256  // minimum duty_period per pwm dac (DAC reg: 8bit ) 

//command:
int main(int argc, char** argv)
{
    const char* imagepath = argv[1];
	  int vid = -1;
	  int w, h, s;
    int off_x,off_y;
    VideoCapture video;
	  float lastdegree_x=90,lastdegree_y=90;
	  float offset_x_last=0,offset_y_last=0;
	  float offset_x_i=0,offset_y_i=0;
	  float kp_x,ki_x,kd_x,kp_y,ki_y,kd_y,dead_zone_x,dead_zone_y;
	  
	  printf("argc:%d\n",argc);

    if (argc < 2) {
        printf("Usage: %s <video_filename|camera_id> [NEW_DATA:1] [P_POINTS:0]\n", argv[0]);
        return 1;
    }
    
	  if (argc > 2) NEW_DATA = atoi(argv[2]);//1: using sunplus optimized model data
	  if (argc > 3) P_POINTS = atoi(argv[3]);//process point , if >=1 it will show character point in the face
	  	
	  if (argc > 4) kp_x = atof(argv[4]);
	  else kp_x = KP_X;
	  if (argc > 5) ki_x = atof(argv[5]);
	  else ki_x = KI_X;
	  if (argc > 6) kd_x = atof(argv[6]);
	  else kd_x = KD_X;
	  if (argc > 7) kp_y = atof(argv[7]);
	  else kp_y = KP_Y;
	  if (argc > 8) ki_y = atof(argv[8]);
	  else ki_y = KI_Y;
	  if (argc > 9) kd_y = atof(argv[9]);
	  else kd_y = KD_Y;		
	  if (argc > 10) dead_zone_x = atof(argv[10]);
	  else dead_zone_x = DEAD_ZONE_X;	
	  if (argc > 11) dead_zone_y = atof(argv[11]);
	  else dead_zone_y = DEAD_ZONE_Y;		  		  	  	
	  	
	  printf("NEW_DATA:%d P_POINTS:%d\n", NEW_DATA, P_POINTS);

#if 0//for debug using
    printf("We have %d arguments:\n", argc);
    for (int i = 0; i < argc; ++i) {
        printf("[%d] %s\n", i, argv[i]);
    }    
#endif

    if (isdigit(imagepath[0]) && !imagepath[1]) {
        vid = imagepath[0] - '0';
        if (!video.open(vid)) {
            return 2;
        }
    }

	  cv::Mat cv_img;
	  if (vid >= 0)
	  	video >> cv_img;
	  else
	  	cv_img = imread(imagepath);
	  imshow(imagepath, cv_img);
	  w = cv_img.cols;
	  h = cv_img.rows;
  	s = (w > h) ? w : h;
  	printf("IMG_SIZE: %d x %d\n", w, h);
  	waitKey(3);
    printf("set start 2 %s\n",__TIME__);

	/** open pwm0/pwm1 ****/
    ofstream ofile("/sys/class/pwm/pwmchip0/export",ios::out);
    ofile<<0<<endl;
    ofile<<1<<endl;
    ofile.close();
	
	/**** pwm0 period set ****/
    ofstream ofile0_period("/sys/class/pwm/pwmchip0/pwm0/period",ios::out);
    if(!ofile0_period)
     {
          printf(" open ofile0_period failed \n");
          exit(1);
     } 
    ofile0_period<<PWM_CYCLE<<endl;
    ofile0_period.close();
	/**** pwm1 period set ****/
    ofstream ofile1_period("/sys/class/pwm/pwmchip0/pwm1/period",ios::out);
     if(!ofile1_period)
      {
           printf(" open ofile1_period\n");
           exit(1);
      } 
    ofile1_period<<PWM_CYCLE<<endl;
    ofile1_period.close();
	/**** pwm0 duty set ****/
    ofstream ofile0_duty("/sys/class/pwm/pwmchip0/pwm0/duty_cycle",ios::out);
     if(!ofile0_duty)
      {
           printf(" open ofile0_duty\n");
           exit(1);
      } 
	  ofile0_duty<<(MOTOR_CENTER_PERIOD)<<endl;//for motor, 1500us period is in the center position
	/**** pwm1 duty set ****/
    ofstream ofile1_duty("/sys/class/pwm/pwmchip0/pwm1/duty_cycle",ios::out);
     if(!ofile1_duty)
      {
           printf(" open ofile1_duty\n");
           exit(1);
      } 
	  ofile1_duty<<(MOTOR_CENTER_PERIOD)<<endl;//for motor, 1500us period is in the center position
	/**** pwm0 enable set ****/
    ofstream ofile0_enable("/sys/class/pwm/pwmchip0/pwm0/enable",ios::out);
     if(!ofile0_enable)
      {
           printf(" open ofile0_enablefailed \n");
           exit(1);
      } 
    ofile0_enable<<1<<endl;
    ofile0_enable.close();
	/**** pwm1 enable set ****/
    ofstream ofile1_enable("/sys/class/pwm/pwmchip0/pwm1/enable",ios::out);
     if(!ofile1_enable)
      {
           printf(" open ofile1_enable failed \n");
           exit(1);
      } 
    ofile1_enable<<1<<endl;
    ofile1_enable.close();

#ifdef SHOW_FPS
    struct timeval tv1, tv2;
    int n = 0; // frames
    gettimeofday(&tv1, NULL); // start
#endif
    
    waitKey(0);//wait push any key of keyboard to start dtection

    while (1) {
      flip(cv_img, cv_img, 1);
    	//char ss[20];
	    std::vector<Bbox> finalBbox;
	    mtcnn mm;
	    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(cv_img.data, ncnn::Mat::PIXEL_BGR, cv_img.cols, cv_img.rows);

	    mm.detect(ncnn_img, finalBbox);
        for(vector<Bbox>::iterator it=finalBbox.begin(); it!=finalBbox.end();it++){
            if((*it).exist){
#if 1
				// draw a box around face
        rectangle(cv_img, Point((*it).x1, (*it).y1), Point((*it).x2, (*it).y2), Scalar(0,0,255), 2,8,0);
#else
				// draw a line from screen center to face center
				line(cv_img, Point(w/2, h/2), Point(((*it).x1 + (*it).x2) / 2, ((*it).y1 + (*it).y2) / 2), Scalar(0,255,255), 2);
#endif
				// draw face landmarks (eyes, nose, mouth ...)
				if (P_POINTS) for(int num=0;num<5;num++)circle(cv_img,Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,Scalar(0,255,255), -1);
				// output offset of face center from screen center
				printf("[%4d, %4d]\n", ((*it).x1 + (*it).x2 - w) * 100 / s, ((*it).y1 + (*it).y2 - h) * 100 / s);
				//putText(cv_img, ss, Point(w/2-110, h/2+10), FONT_HERSHEY_COMPLEX, 1, Scalar(0,0,255), 2);
				        //x,y axis position error caculation
                float offset_x = (((*it).x1  + ((*it).x2-(*it).x1)/2.0) / w - 0.5) * 2;
                float offset_y = (((*it).y1  + ((*it).y2-(*it).y1)/2.0) / h - 0.5) * 2;
                
                if (abs(offset_x) > dead_zone_x)
                {              
                    offset_x_i += offset_x; 
                    float temp_x_degree = (offset_x*kp_x + offset_x_i*ki_x + (offset_x - offset_x_last)*kd_x) + lastdegree_x;
                    if (temp_x_degree < 2)
                        temp_x_degree = 2;
                    else if(temp_x_degree > 178)
                        temp_x_degree = 178;

                    int duty_cur = (MOTOR_DOWN_LIMIT_PERIOD+lastdegree_x*(MOTOR_UP_LIMIT_PERIOD-MOTOR_DOWN_LIMIT_PERIOD)/180);
                    
                    int duty_period = (MOTOR_DOWN_LIMIT_PERIOD+temp_x_degree*(MOTOR_UP_LIMIT_PERIOD-MOTOR_DOWN_LIMIT_PERIOD)/180);

                    printf(" X [%f]  lastdegree_x=%f  tardegree_x=%f  duty=%d \n",offset_x,lastdegree_x,temp_x_degree,duty_period);
                    lastdegree_x = temp_x_degree;
                    offset_x_last = offset_x;

                    ofile0_duty<<(duty_period)<<endl;
                  
                }

                if (abs(offset_y) > dead_zone_y)
                {            
                    offset_x_i += offset_x; 
                    float temp_y_degree = (offset_y*kp_y + offset_y_i*ki_x + (offset_y - offset_y_last)*kd_y) + lastdegree_y;                    
                    if (temp_y_degree < 2)
                       temp_y_degree = 2;
                    else if(temp_y_degree > 178)
                       temp_y_degree = 178;

                    int duty_cur = (MOTOR_DOWN_LIMIT_PERIOD+lastdegree_y*(MOTOR_UP_LIMIT_PERIOD-MOTOR_DOWN_LIMIT_PERIOD)/180);

                    int duty_period = (MOTOR_DOWN_LIMIT_PERIOD+temp_y_degree*(MOTOR_UP_LIMIT_PERIOD-MOTOR_DOWN_LIMIT_PERIOD)/180);

                    printf(" Y [%f]  lastdegree_y=%f  tardegree_y=%f  duty=%d \n",offset_y,lastdegree_y,temp_y_degree,duty_period);//hugo debug
                    lastdegree_y = temp_y_degree;
                    offset_y_last = offset_y;//hugo debug

                    ofile1_duty<<(duty_period)<<endl;                
                }
#if 0//for debug using
            	  off_x = ((*it).x1 + (*it).x2 - w) * 100 / s;
                off_y = ((*it).y1 + (*it).y2 - h) * 100 / s;
                printf("off_x off_y : [%d, %d]\n",off_x,off_y);
                printf("\n");
#endif                
            }
        }
		imshow(imagepath, cv_img);
		waitKey(2);

		if (vid >= 0)
			video >> cv_img;
		else
			cv_img = imread(imagepath);
#ifdef SHOW_FPS
		n++;
		gettimeofday(&tv2, NULL);
		printf("%5.2f\n", n / getElapse(&tv1, &tv2));
#endif
	}
	  ofile1_duty.close();
    ofile0_duty.close();
    return 0;
}
