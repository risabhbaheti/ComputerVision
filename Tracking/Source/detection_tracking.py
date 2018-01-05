import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

def skeleton_tracker_camshift(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter, c+w/2,r+h/2)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you

    # initialize the tracker
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        (c,r,w,h) = track_window
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        #Display code
        #frame = cv2.polylines(frame,[pts],True, 255,2)
        #frame = cv2.circle(frame, (c+w/2, r+h/2), 2, (0, 255, 0), -1)
        #cv2.imshow("mean_shift",frame)
        #k = cv2.waitKey(60) & 0xff
        #cv2.destroyAllWindows()

        # write the result to the output file
        
        output.write("%d,%d,%d\n" % (frameCounter, c+w/2,r+h/2)) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()
    
def skeleton_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.2,
                       minDistance = 3,
                       blockSize = 3 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    frameCounter = 0

    # read first frame
    ret ,frame = v.read()
    c,r,w,h = detect_one_face(frame)
    #cv2.imshow("mean_shift",frame[r-20:r+h+20, c-20:c+w+20])
    #k = cv2.waitKey(6000) & 0xff
    
    if ret == False:
        return
    c,r,w,h = detect_one_face(frame)
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray[r:r+h, c:c+w], mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame)

    # detect face in first frame
    
    
    p2 = p0
    count =0
    point1 =0
    point2 =0
    for i in p0:
        x=i[0][0] +c
        y=i[0][1] +r
        point1 = point1 +x
        point2 = point2 +y 
        p2[count][0][0]=x
        p2[count][0][1]=y
        count = count +1

    # Write track point for first frame
    output.write("%d,%d,%d\n" % (0,int(point1/count),int(point2/count))) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        point1=0.0
        point2=0.0
        count =0
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            point1 = point1 +a
            point2 = point2 +b
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 1)
            frame = cv2.circle(frame,(a,b),1,color[i].tolist(),-1)
            img = cv2.add(frame,mask)
            count =count+1
            #cv2.arrowedLine(img, p0, p1, (255, 0, 0), tipLength=0.5)
        point1 =point1/count
        point2 =point2/count
        frame = cv2.circle(frame,(int(point1),int(point2)),5,(255,0,0),-1)   
        #cv2.imshow("frame",img)
        #cv2.waitKey(200)&0xff
        #cv2.destroyAllWindows()


        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        #print p0
        #print p1

        # use the tracking result to get the tracking point (pt):
        # if you track a rect (e.g. face detector) take the mid point,
        # if you track particles - take the weighted average
        # the Kalman filter already has the tracking point in the state vector

        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter,point1,point2)) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()
    
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

def skeleton_tracker_particle(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter, c+w/2,r+h/2)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1


    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    n_particles = 200
    init_pos = np.array([c + w/2.0,r + h/2.0], int)
    #hist_bp = cv2.calcBackProject([roi_hist],[0],roi_hist,[0,180],1)
    # initialize the tracker
    particles = np.ones((n_particles, 2), int) * init_pos
    #f0 = particleevaluator(hist_bp, particles) * np.ones(n_particles) # Evaluate appearance model
    weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)
    stepsize = 10
    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")
        # Clip out-of-bounds particles
        particles = particles.clip(np.zeros(2), np.array((frame.shape[1],frame.shape[0]))-1).astype(int)
        #For Display
        #for par in particles:
        #    frame = cv2.circle(frame, (par[0], par[1]), 2, (0, 255, 0), -1)
        #cv2.imshow("mean_shift",frame)
        #k = cv2.waitKey(60) & 0xff
        #cv2.destroyAllWindows()
        f = particleevaluator(dst, particles.T) # Evaluate particles
        weights = np.float32(f.clip(1))             # Weight ~ histogram response
        weights /= np.sum(weights)                  # Normalize w
        pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average
        output.write("%d,%d,%d\n" % (frameCounter, pos[0],pos[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
        
        if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
            particles = particles[resample(weights),:]  # Resample particles according to weights
        

    output.close()

def skeleton_tracker_kalman(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter, c+w/2,r+h/2)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
    # initialize the tracker
    kf = cv2.KalmanFilter(4,2,0)
    # or: particles = np.ones((n_particles, 2), int) * initial_pos
    kf.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
    kf.measurementMatrix = 1. * np.eye(2, 4)
    kf.processNoiseCov = 1e-5 * np.eye(4, 4)
    kf.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kf.errorCovPost = 1e-1 * np.eye(4, 4)
    kf.statePost = state
    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break
        c,r,w,h=detect_one_face(frame)
        posterior = kf.predict()
        if(c!=0 or r!=0 or w!=0 or h!=0):
            measurement = np.array([c+w/2, r+h/2], dtype='float64')
            posterior = kf.correct(measurement)
        x = posterior[0]
        y = posterior[1] 
        #Display code
        #frame = cv2.rectangle(frame,(c,r),(c+w,r+h),255,2)
        #frame = cv2.circle(frame, (c+w/2, r+h/2), 2, (0, 255, 0), -1)
        #frame = cv2.circle(frame, (int(x),int(y)), 2, (255, 0, 0), -1)
        #cv2.imshow("mean_shift",frame)
        #k = cv2.waitKey(60) & 0xff
        #cv2.destroyAllWindows()
        output.write("%d,%d,%d\n" % (frameCounter, x,y)) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()

if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        skeleton_tracker_camshift(video, "output_camshift.txt")
    elif (question_number == 2):
        skeleton_tracker_particle(video, "output_particle.txt")
    elif (question_number == 3):
        skeleton_tracker_kalman(video, "output_kalman.txt")
    elif (question_number == 4):
        skeleton_tracker(video, "output_of.txt")


