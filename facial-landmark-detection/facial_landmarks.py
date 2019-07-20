# import the necessary packages
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import math
from collections import Counter

#eyeQ = Queue.Queue()
eyeStateArr = [] 
mouthStateArr = [] 
FILE_OUTPUT = 'output.avi'

def actiondetect(cordDict,framecount):
    eyeState = eyeStateDetection(cordDict,framecount)
    mouthState = mouthStateDetection(cordDict,framecount)
    
    return eyeState+','+mouthState
    
    pass

def  mouthStateDetection(cordDict,framecount):
    global mouthStateArr
    mouthDistRatio = distance(cordDict[62],cordDict[66])/distance(cordDict[60],cordDict[54])
    if(mouthDistRatio > 0.12):
        mouthState = 'open'
    else:
        mouthState = 'closed'
        
    mouthStateArr.append(mouthState)
    if len(mouthStateArr) >=5:
        mouthStateArr = mouthStateArr[-5:]
        countDict = Counter(mouthStateArr)
        if countDict.get('open'):
            if countDict.get('open') >=2:
                return 'talking'
            else:
                return ''
        else:
            return ''
    return ''
    pass

def eyeStateDetection(cordDict,framecount):
    global eyeStateArr
    eyeDistRatio = distance(cordDict[37],cordDict[41])/distance(cordDict[36],cordDict[39])
    if(eyeDistRatio > 0.22):
        eyeState = 'open'
    else:
        eyeState = 'closed'
    eyeStateArr.append(eyeState)
    
    if len(eyeStateArr) >=5:
        
        #print(Counter(eyeStateArr))
        #Only retain the last 5 states
        eyeStateArr = eyeStateArr[-5:]
        
        countDict = Counter(eyeStateArr)
        
        #count the occurances of slee
        if countDict.get('closed') != None and countDict.get('closed') > 2:
            return 'sleeping'
        elif countDict.get('closed') != None and countDict.get('closed') > 0:
            if 'closed' in eyeStateArr[-3:]:
                return 'blinking'
            else:
                return 'awake'
        else:
            return 'awake'
    return 'awake'

def distance(cord1,cord2):
    (x1,y1) = cord1
    (x2,y2) = cord2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
 
# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] camera sensor warming up...")
#vs = VideoStream(src=1).start()
video_capture = cv2.VideoCapture(0)
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

framecount = 0

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# Define the fps to be equal to 10. Also frame size is passed.
ret, frame = video_capture.read()
 
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'MJPG'), 20, (800,600))

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 600 pixels, and convert it to
	# grayscale
	#frame = vs.read()
    #Capture frame-by-frame
    ret, frame = video_capture.read()
    framecount = framecount+1
    frame = imutils.resize(frame, width=800, height=600)
    #out.write(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
       
    # loop over the face detections
    for rect in rects:
		# compute the bounding box of the face and draw it on the
		# frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        #cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)
 
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
 
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw each of them
        px = None
        py = None
        cordDict = {}
        for (i, (x, y)) in enumerate(shape):
            cordDict[i]=(x,y)
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
            #Uncomment the below line incase you want to see the number of the point detected
            #cv2.putText(frame, str(i), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
            #Connect coordinates to join lines between the landmark
            if (px!=None or py!=None) and i not in [17,22,27,36,42,48,60] :
                cv2.line(frame,(x,y),(px,py),(0, 0, 255),1) 
                pass
            px=x
            py=y
        #Mapping the gaps
        cv2.line(frame,cordDict[17],cordDict[21],(0, 0, 255),1)
        cv2.line(frame,cordDict[22],cordDict[26],(0, 0, 255),1)
        cv2.line(frame,cordDict[36],cordDict[41],(0, 0, 255),1)
        cv2.line(frame,cordDict[42],cordDict[47],(0, 0, 255),1)
        cv2.line(frame,cordDict[30],cordDict[35],(0, 0, 255),1)
        cv2.line(frame,cordDict[17],cordDict[21],(0, 0, 255),1)
        cv2.line(frame,cordDict[48],cordDict[59],(0, 0, 255),1)
        cv2.line(frame,cordDict[60],cordDict[67],(0, 0, 255),1)
        action = actiondetect(cordDict,framecount)
        # check to see if a face was detected, and if so, draw the total
        text = "face detected | state: {}".format(action)
    
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
    			1, (0, 0, 255), 2)      
        
    
    # Saves for video
    out.write(frame)
    
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
out.release()
video_capture.release()
cv2.destroyAllWindows()