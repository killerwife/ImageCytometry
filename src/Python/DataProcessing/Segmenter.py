import cv2
outputFolder = 'D:\\BigData\\cellinfluid\\2000fps_40x(0.55)_shut30000_0.2%htc_31umQ2.5_C001H001S0001\\Frames\\'
vidcap = cv2.VideoCapture('D:\\BigData\\cellinfluid\\2000fps_40x(0.55)_shut30000_0.2%htc_31umQ2.5_C001H001S0001\\2000fps_40x(0.55)_shut30000_0.2%htc_31umQ2.5_C001H001S0001.avi')
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite(outputFolder + ("frame%d.png" % count), image)     # save frame as JPEG file
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1