import cv2
import time
import numpy as np
import RBC_detection_full_script as detect


def showImage():
    image = cv2.imread("D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\1-50\\video2359_0001.tiff")
    cv2.imshow("Over the Clouds", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # filepath = 'D:\\BigData\\cellinfluid\\output.mp4'
    # cap = cv2.VideoCapture('')
    filepath = 'D:\\BigData\\cellinfluid\\Video_S1_Dylan_RBC_deformability.mp4'
    cap = cv2.VideoCapture(filepath)

    # background = detect.get_backgroundFromVideo(filepath)
    # background = cv2.imread('backgroundOldVid.png')
    # background = cv2.imread('background.jpg')
    # backgroundGray = detect._get_background('D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\501-2992','video2359_%u.tiff',1000,1500)
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, 20, 40)

            cv2.imshow('frame', edges)
            time.sleep(1)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            cv2.imwrite('doubleEdges.png', edges)
            break
        else:
            break

    # backgroundGray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # backgroundFile = open('outputBackground.txt', "w")
    # np.savetxt(backgroundFile, backgroundGray)
    # count = 1
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #
    #     if ret:
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         # frameFile = open('outputFrame.txt', "w")
    #         # np.savetxt(frameFile, gray)
    #         gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #         cv2.imshow('original', gray)
    #         # cv2.imwrite('originalOldVid.png', gray)
    #         subtracted = detect._background_subtraction(gray, backgroundGray)
    #         edges = detect._cany_edge_detection(gray)
    #         cv2.imshow('frame', subtracted)
    #         # cv2.imwrite('frameOldVid.png', subtracted)
    #         cv2.imshow('edges', edges)
    #         # cv2.imwrite('edgesOldVid.png', edges)
    #         edges = cv2.Canny(subtracted, 100,150)
    #         cv2.imshow('edgesOwn', edges)
    #         # cv2.imwrite('edgesOwnOldVid.png', edges)
    #         # time.sleep(1)
    #         # file = open('outputFile.txt', "w")
    #         # np.savetxt(file, subtracted)
    #         # if count == 1:
    #         #     break
    #         # count += 1
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break

    # cv2.imshow('frame', background)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
