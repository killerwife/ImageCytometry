// DetectionOfCells.cpp : Defines the entry point for the console application.
//

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudabgsegm.hpp>
#include "opencv2\cudaobjdetect.hpp"
#include "opencv2\objdetect.hpp"
#include <opencv2\features2d.hpp>

#include <ctime>
#include <chrono>
#include <iostream>
#include <unordered_map>

#include "fileFinder.h"

void detection(std::string model, std::string file, bool GPU, std::string save = "")
{
    //window  
    //cv::namedWindow("origin");

    //load image  
    cv::Mat img = cv::imread(file);
    cv::Mat grayImg; //adaboost detection is gray input only.  
    cvtColor(img, grayImg, CV_BGR2GRAY);

    //load xml file  
    std::string trainface = model;

    //declaration  
    cv::CascadeClassifier ada_cpu;

    if (!(ada_cpu.load(trainface)))
    {
        printf(" cpu ada xml load fail! \n");
        return;
    }

    //if (!(ada_gpu.load(trainface)))
    //{
    //    printf(" gpu ada xml load fail! \n");
    //    return;
    //}

    //////////////////////////////////////////////  
    //cpu case face detection code  
    std::vector< cv::Rect > faces;
    //Atime = cv::getTickCount();
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    ada_cpu.detectMultiScale(grayImg, faces,1.01,3,0,cv::Size(25,25),cv::Size(40,40));
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "CPU finished computation at " << std::ctime(&end_time)
        << "elapsed time: " << elapsed_seconds.count() << "s\n";
    //Btime = cv::getTickCount();
    //TakeTime = (Btime - Atime) / cv::getTickFrequency();
    //printf("detected face(cpu version) = %d / %lf sec take.\n", faces.size(), TakeTime);
    if (faces.size() >= 1)
    {
        for (int ji = 0; ji < faces.size(); ++ji)
        {
            rectangle(img, faces[ji], CV_RGB(0, 0, 255), 4);
        }
    }

    if (GPU)
    {
        /////////////////////////////////////////////  
        //gpu case face detection code  
        cv::Ptr<cv::cuda::CascadeClassifier> ada_gpu = cv::cuda::CascadeClassifier::create(model);

        cv::cuda::GpuMat faceBuf_gpu;
        cv::cuda::GpuMat GpuImg;

        GpuImg.upload(grayImg);
        start = std::chrono::system_clock::now();
        ada_gpu->detectMultiScale(GpuImg, faceBuf_gpu);
        end = std::chrono::system_clock::now();

        elapsed_seconds = end - start;
        std::chrono::system_clock::to_time_t(end);

        std::cout << "GPU finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
        //printf("detected face(gpu version) =%d / %lf sec take.\n", detectionNumber, TakeTime);

        //    std::vector<cv::Rect> faces;
        ada_gpu->convert(faceBuf_gpu, faces);

        for (int i = 0; i < faces.size(); ++i)
            cv::rectangle(img, faces[i], cv::Scalar(255));
    }


    /////////////////////////////////////////////////  
    //result display  
    if (!save.empty())
        imwrite(save, img);
    else
    {
        cv::imshow("Output", img);
        cv::waitKey(0);
    }
}

void processImage(int /*h*/, void*)
{
    std::vector<std::vector<cv::Point> > contours;
    std::string pathInputOriginal = "D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\all";
    cv::Mat originalInputImage = cv::imread(pathInputOriginal + "\\" + "video2359_0151.tiff", cv::IMREAD_GRAYSCALE);
    int sliderPos = 70;
    cv::Mat bimage = originalInputImage >= sliderPos;

    findContours(bimage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    cv::Mat cimage = cv::Mat::zeros(bimage.size(), CV_8UC3);

    for (size_t i = 0; i < contours.size(); i++)
    {
        size_t count = contours[i].size();
        if (count < 6)
            continue;

        cv::Mat pointsf;
        cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
        cv::RotatedRect box = cv::fitEllipse(pointsf);

        if (MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height) * 30)
            continue;
        drawContours(cimage, contours, (int)i, cv::Scalar::all(255), 1, 8);

        ellipse(cimage, box, cv::Scalar(0, 0, 255), 1, CV_AA);
        ellipse(cimage, box.center, box.size*0.5f, box.angle, 0, 360, cv::Scalar(0, 255, 255), 1, CV_AA);
        cv::Point2f vtx[4];
        box.points(vtx);
        for (int j = 0; j < 4; j++)
            line(cimage, vtx[j], vtx[(j + 1) % 4], cv::Scalar(0, 255, 0), 1, CV_AA);
    }

    imshow("result", cimage);
    cv::waitKey(0);
}

void scripts(int id)
{
    switch (id)
    {
        case 0: // create background from all images
        {
            FileFinder finder;
            std::vector<std::string> filenames;
            std::string path = "D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\all";
            finder.GetFileNames(path, filenames);

            cv::Mat frame; //current frame
            cv::Mat greyscaleFrame;
            cv::Mat fgMaskMogCPU;
            cv::cuda::GpuMat fgMaskMOG; //fg mask fg mask generated by MOG2 method
            cv::Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
            cv::Ptr<cv::BackgroundSubtractor> pMOG; //MOG Background subtractor
            cv::Ptr<cv::BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
            pMOG = cv::cuda::createBackgroundSubtractorMOG(); //MOG approach
            pMOG2 = cv::createBackgroundSubtractorMOG2(); //MOG2 approach
            for (std::string& filename : filenames)
            {
                frame = cv::imread(path + "\\" + filename);
                cv::cvtColor(frame, greyscaleFrame, CV_BGR2GRAY);
                cv::cuda::GpuMat imageGpu;
                imageGpu.upload(greyscaleFrame);
                pMOG->apply(imageGpu, fgMaskMOG);
                pMOG2->apply(frame, fgMaskMOG2);
            }
            fgMaskMOG.download(fgMaskMogCPU);
            cv::cuda::GpuMat backgroundImageGPU;
            cv::Mat backgroundImage;
            pMOG->getBackgroundImage(backgroundImageGPU);
            backgroundImageGPU.download(backgroundImage);
            cv::imwrite("outputGPUBackground.png", backgroundImage);
            pMOG2->getBackgroundImage(backgroundImage);
            cv::imwrite("outputCPUBackground.png", backgroundImage);
            break;
        }
        case 1: // subtract evaluated background from all images
        {
            FileFinder finder;
            std::vector<std::string> filenamesInput;
            std::string pathInput = "D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\all";
            finder.GetFileNames(pathInput, filenamesInput);

            std::string pathOutput = "D:\\BigData\\cellinfluid\\subtractedBackgrounds\\all";
            cv::Mat background = cv::imread("outputCPUBackground.png");
            for (std::string& filename : filenamesInput)
            {
                cv::Mat mat = cv::imread(pathInput + "\\" + filename);
                mat -= background;
                cv::imwrite(pathOutput + "\\" + filename, mat);
            }
            break;
        }
        case 2:
        {
            std::string pathInput = "D:\\BigData\\cellinfluid\\subtractedBackgrounds\\all";
            std::string imageName = "video2359_2378.tiff";
            cv::Mat imageWithoutBackground = cv::imread(pathInput + "\\" + imageName);
            imageWithoutBackground *= 5;
            cv::imwrite("NoBackground.png", imageWithoutBackground);
            cv::waitKey(0);
            break;
        }
        case 3: // without background subtraction - raw images
        {
            std::string pathInput = "D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\all";
            detection("cascadeWithoutSubtraction3Specialparams.xml", pathInput + "\\" + "video2359_0151.tiff", false, "OutputNoSubtraction3.png");
            break;
        }
        case 4: // with background subtraction
        {
            std::string pathInput = "D:\\BigData\\cellinfluid\\subtractedBackgrounds\\all";
            detection("cascadeWithBackgroundSubtraction.xml", pathInput + "\\" + "video2359_0151.tiff", false, "OutputSubtraction.png");
            break;
        }
        case 5: // canny edge detection
        {
            std::string pathInput = "D:\\BigData\\cellinfluid\\subtractedBackgrounds\\all";
            std::string pathInputOriginal = "D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\all";
            cv::Mat originalInputImage = cv::imread(pathInputOriginal + "\\" + "video2359_0151.tiff");
            cv::Mat inputImage = cv::imread(pathInput + "\\" + "video2359_0151.tiff");
            cv::Mat edges;
            int ratio = 3;
            int threshold = 20;
            cv::Canny(inputImage, edges, threshold, threshold * ratio);
            cv::imwrite("originalInputImageCanny.png", originalInputImage);
            cv::imwrite("cannySubtractedBackground.png", edges);
            cv::waitKey(0);
            break;
        }
        case 6: // circle detection
        {
            std::vector<cv::Vec3f> circles;
            std::string pathInput = "D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\all";
            cv::Mat inputImage = cv::imread(pathInput + "\\" + "video2359_0151.tiff", CV_LOAD_IMAGE_GRAYSCALE);
            /// Apply the Hough Transform to find the circles
            cv::HoughCircles(inputImage, circles, CV_HOUGH_GRADIENT, 1, inputImage.rows / 8, 15, 15, 0, 30);

            /// Draw the circles detected
            for (size_t i = 0; i < circles.size(); i++)
            {
                cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                int radius = cvRound(circles[i][2]);
                // circle center
                cv::circle(inputImage, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
                // circle outline
                cv::circle(inputImage, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
            }

            cv::imshow("Image", inputImage);
            cv::waitKey(0);
        }
        case 7:
        {
            std::string pathInputOriginal = "D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\all";
            std::string pathInputSubtracted = "D:\\BigData\\cellinfluid\\subtractedBackgrounds\\all";
            cv::Mat originalInputImage = cv::imread(pathInputOriginal + "\\" + "video2359_0151.tiff");
            cv::Mat subtractedInputImage = cv::imread(pathInputSubtracted + "\\" + "video2359_0151.tiff");
            cv::Ptr<cv::MSER> blobsDetector = cv::MSER::create();
            std::vector<cv::KeyPoint> keypoints;
            blobsDetector->detect(subtractedInputImage, keypoints);
            for (size_t i = 0; i < keypoints.size(); ++i)
                circle(originalInputImage, keypoints[i].pt, 4, cv::Scalar(255, 0, 255), -1);
            cv::imwrite("FeatureDetectorSubtractedMSER.png", originalInputImage);
            cv::waitKey(0);
            break;
        }
    }
}

#define PATH_MAX 512

struct ObjectPos
{
    float x;
    float y;
    float width;
    bool found;    /* for reference */
};

void cascadePerformance(int argc, char* argv[])
{
    std::string classifierName;
    char* inputName = (char*)"";
    float maxSizeDiff = 0.5F;
    float maxPosDiff = 0.3F;
    double scaleFactor = 1.25;
    int minNeighbors = 3;
    cv::Size minSize;
    cv::Size maxSize;
    bool equalize = false;;
    bool saveDetected = false;
    FILE* info;
    cv::CascadeClassifier cascade;
    double totaltime = 0.0;

    if (argc <= 2)
    {
        std::cout << "Aplikacia je urcena na vyhodnotenie vykonosti natrenovaneho detektora" << std::endl;
        std::cout << "Pouzitie: " << std::endl;
        std::cout << "  -classifier <classifier_directory_name>" << std::endl;
        std::cout << "  -input <collection_file_name>" << std::endl;
        std::cout << "  [-maxSizeDiff <max_size_difference = " << maxSizeDiff << ">]" << std::endl;
        std::cout << "  [-maxPosDiff <max_position_difference = " << maxPosDiff << ">]" << std::endl;
        std::cout << "  [-sf <scale_factor = " << scaleFactor << ">]" << std::endl;
        std::cout << "  [-minNeighbors <min_number_neighbors_for_each_candidate = " << minNeighbors << " >]" << std::endl;
        std::cout << "  [-minSize <min_possible_object_size> Example: 32x32 (Width * Height)]" << std::endl;
        std::cout << "  [-maxSize <max_possible_object_size> Example: 64x64 (Width * Height)]" << std::endl;
        std::cout << "  [-equalize <histogram_equalization: " << (equalize ? "True" : "False") << ">]" << std::endl; // ??
        std::cout << "  [-save <save_detection: " << (saveDetected ? "True" : "False") << ">]" << std::endl;
        return;
    }

    for (int i = 2; i < argc; i++)
    {
        if (!strcmp(argv[i], "-classifier"))
        {
            classifierName = argv[++i];
        }
        else if (!strcmp(argv[i], "-input"))
        {
            inputName = argv[++i];
        }
        else if (!strcmp(argv[i], "-maxSizeDiff"))
        {
            float tmp = (float)atof(argv[++i]);
            if (tmp >= 0 && tmp <= 1)
                maxSizeDiff = tmp;
        }
        else if (!strcmp(argv[i], "-maxPosDiff"))
        {
            float tmp = (float)atof(argv[++i]);
            if (tmp >= 0 && tmp <= 1)
                maxPosDiff = tmp;
        }
        else if (!strcmp(argv[i], "-sf"))
        {
            double tmp = atof(argv[++i]);
            if (tmp > 1)
                scaleFactor = tmp;
        }
        else if (!strcmp(argv[i], "-minNeighbors"))
        {
            minNeighbors = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-minSize"))
        {
            sscanf(argv[++i], "%ux%u", &minSize.width, &minSize.height);
        }
        else if (!strcmp(argv[i], "-maxSize"))
        {
            sscanf(argv[++i], "%ux%u", &maxSize.width, &maxSize.height);
        }
        else if (!strcmp(argv[i], "-equalize"))
        {
            equalize = true;
        }
        else if (!strcmp(argv[i], "-save"))
        {
            saveDetected = true;
        }
        else
            std::cerr << "WARNING: Neznama volba " << argv[i] << std::endl;
    }

    if (!cascade.load(classifierName))
    {
        std::cerr << "ERROR: Nemozem nacitat klasifikator" << std::endl;
        return;
    }

    char fullname[PATH_MAX];
    char detfilename[PATH_MAX];
    char* filename;
    char detname[] = "det-";

    // skopiruje do fullname cestu
    strcpy(fullname, inputName);
    // do filename vlozi smernik na posledny vyskyt '\\'
    filename = strrchr(fullname, '\\');
    if (filename == NULL)
    {
        // do filename vlozi smernik na posledny vyskyt znaku '/'
        filename = strrchr(fullname, '/');
    }
    if (filename == NULL)
    {
        filename = fullname;
    }
    else
    {
        filename++;
    }

    info = fopen(inputName, "r");
    if (info == NULL)
    {
        std::cerr << "ERROR: Nemozem otvorit vstupny subor" << std::endl;
        return;
    }

    std::cout << "Parametre: " << std::endl;
    std::cout << "Classifier: " << classifierName << std::endl;
    std::cout << "Input: " << inputName << std::endl;
    std::cout << "maxSizeDiff: " << maxSizeDiff << std::endl;
    std::cout << "maxPosDiff: " << maxPosDiff << std::endl;
    std::cout << "sf: " << scaleFactor << std::endl;
    std::cout << "minNeighbors: " << minNeighbors << std::endl;
    std::cout << "minSize: " << minSize << std::endl;
    std::cout << "maxSize: " << maxSize << std::endl;
    std::cout << "equalize: " << (equalize ? "True" : "False") << std::endl;
    std::cout << "save: " << (saveDetected ? "True" : "False") << std::endl;

    cv::Mat image, grayImage;
    int hits, missed, falseAlarms;
    int totalHits = 0, totalMissed = 0, totalFalseAlarms = 0, totalObjects = 0;
    int found;
    int refcount;

    std::cout << "+================================+======+======+======+=======+" << std::endl;
    std::cout << "|            File Name           | Hits |Missed| False|Objects|" << std::endl;
    std::cout << "+================================+======+======+======+=======+" << std::endl;
    while (!feof(info))
    {
        if (fscanf(info, "%s %d", filename, &refcount) != 2 || refcount < 0)
            break;

        image = cv::imread(fullname, cv::IMREAD_COLOR);
        if (image.empty())
        {
            std::cerr << "WARNING: Obrazok sa nepodarilo nacitat: " << fullname << std::endl;
            continue;
        }

        // nacitanie suradnic objektov vo vstupnom subore
        int x, y, w, h;
        ObjectPos tmp;
        std::vector<ObjectPos> ref;
        for (int i = 0; i < refcount; i++)
        {
            if (fscanf(info, "%d %d %d %d", &x, &y, &w, &h) != 4)
            {
                std::cerr << "ERROR: Nespravny format vstupneho suboru" << std::endl;
                return;
            }

            // vypocet stredu obdlznika
            tmp.x = 0.5F * w + x;
            tmp.y = 0.5F * h + y;
            // vypocet priemernej dlzky strany
            tmp.width = 0.5F * (w + h);
            tmp.found = false;
            ref.push_back(tmp);

            if (saveDetected)
                rectangle(image, cv::Rect(x, y, w, h), CV_RGB(0, 255, 0));
        }

        // spustenie detekcie na nacitanom obrazku
        cvtColor(image, grayImage, CV_BGR2GRAY);
        // ekvalizáciu histogramu ak je zadane -norm
        if (equalize)
        {
            cv::Mat temp;
            equalizeHist(grayImage, temp);
            grayImage = temp;
        }

        std::vector<cv::Rect> objects;
        totaltime -= (double)cvGetTickCount();
        cascade.detectMultiScale(grayImage, objects, scaleFactor, minNeighbors, 0, minSize, maxSize);
        totaltime += (double)cvGetTickCount();
        hits = missed = falseAlarms = 0;

        ObjectPos det;
        // meranie vykonosti
        for (int i = 0; i < objects.size(); i++)
        {
            ////PERFORMANCE SCALE
            //tmpw = objects[i].width / (float) 1.1; // sirka zmensena o 10%
            //tmph = objects[i].height / (float) 1.1; // vyska zmensena o 10%

            // vypocet stredu obdlznika
            det.x = 0.5F * objects[i].width + objects[i].x;
            det.y = 0.5F * objects[i].height + objects[i].y;
            // vypocet priemernej dlzky strany
            det.width = 0.5F * (objects[i].width + objects[i].height);

            ////PERFORMANCE SCALE
            //det.width = sqrtf( 0.5F * (tmpw * tmpw + tmph * tmph));		
            //// uplatnenie zmensenia aj do vykresleneho bb
            //objects[i].width = tmpw; // sirka zmensena o 10%
            //objects[i].height = tmph; // vyska zmensena o 10%
            //objects[i].x = (objects[i].x + (objects[i].width - tmpw) / (float)2 );
            //objects[i].y = (objects[i].y + (objects[i].height - tmph) / (float)2 );

            found = 0;
            for (int j = 0; j < refcount; j++)
            {
                float distance = sqrtf((det.x - ref[j].x) * (det.x - ref[j].x) +
                    (det.y - ref[j].y) * (det.y - ref[j].y));

                if ((distance < ref[j].width * maxPosDiff) &&
                    (det.width > ref[j].width - ref[j].width * maxSizeDiff) &&
                    (det.width < ref[j].width + ref[j].width * maxSizeDiff))
                {
                    ref[j].found = 1;
                    found = 1;
                    if (saveDetected)
                        rectangle(image, objects[i], CV_RGB(0, 0, 255), 2);
                }
            }

            if (!found)
                falseAlarms++;

            // ulozenie vysledku detekcie ak je zadane -save
            if (saveDetected && !found)
                rectangle(image, objects[i], CV_RGB(255, 0, 0));
        }

        for (int j = 0; j < refcount; j++)
        {
            if (ref[j].found)
                hits++;
            else
                missed++;
        }

        totalHits += hits;
        totalMissed += missed;
        totalFalseAlarms += falseAlarms;
        totalObjects += objects.size();
        printf("|%32.32s|%6d|%6d|%6d|%7lld|\n", filename, hits, missed, falseAlarms, objects.size());
        std::cout << "+--------------------------------+------+------+------+-------+" << std::endl;
        fflush(stdout);

        if (saveDetected)
        {
            strcpy(detfilename, detname);
            strcat(detfilename, filename);
            strcpy(filename, detfilename);
            imwrite(fullname, image);
        }
    }
    fclose(info);

    printf("|%32.32s|%6d|%6d|%6d|%7d|\n", "Total", totalHits, totalMissed, totalFalseAlarms, totalObjects);
    std::cout << "+--------------------------------+------+------+------+-------+" << std::endl;
    //printf( "Number of stages: %d\n",  );
    //printf( "Number of weak classifiers: %d\n", );
    printf("Celkovy cas detekcie: %g ms\n", totaltime / ((double)cvGetTickFrequency()*1000.));
}

int main(int argc, char* argv[])
{
    scripts(7);
    //processImage(1,nullptr);
    //cascadePerformance(argc, argv);

    return 0;
}

