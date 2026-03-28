import cv2
import mediapipe as mp

class HandDetector:
    
    """
    A class for the MediaPipe Hands solution. This class handles the initialization,
    processing, and coordinate extraction of hand landmarks
    """
    
    def __init__(self, mode=False, max_hands=1, detection_con=0.5, track_con=0.5):
        """
        Initializes the MediaPipe Hands model with specific parameters
        
        :param mode: False, treats the input as a video steam
        :param max_hands: Maximum number of hands to detect simultaneously
        :param detection_con: Minimum confidence threshold for the initial hand
                              detection to be considered succesful
        :param track_on: Minimum confidence threshold for tracking the hand across
                         subsequent frames
        """
        
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        
        # Load the MediaPipe Hands module
        self.mp_hands = mp.solutions.hands
        
        # Initialize the core tracking engine
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        
        # Load the drawing utilities to visualize the hand skeleton
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4,8,12,16,20]   #Thumb, index, middle, ring, little fingers
        
    def find_hands(self, img, draw=True):
        """
        Processes the input image to find hand landmarks and optionally draws
        the skeleton
        
        :param img: The raw BGR image frame captured by OpenCV
        :param draw: Boolean flag to enable/disable drawing the bone connections
        :return: The processed image
        """
        
        # OpenCV captures in BGR, but MediaPipe requires RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Pass the RGB image to the NN for landmark detection
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw the 21 dots and the lines connecting them
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    
        return img
    
    def find_position(self, img, hand_no=0):
        """
        Extracts the precise X and Y pixel coordinates for all 21 landmarks of a
        detected hand
        
        :param img: The current image frame
        :param hand_no: The index of the hand to extract
        :return: A list of lists containing [landmark_id, x_pixel, y_pixel]
        """
        
        self.lm_list = []
        
        # Only if the hands processor succesfully found landmarks
        if self.results.multi_hand_landmarks:
            # Select the specific hand we want to track
            my_hand = self.results.multi_hand_landmarks[hand_no]
            
            # Get the exact dimensions of out video window
            h, w, c = img.shape
            
            # Loop throgh all 21 points
            for id, lm in enumerate(my_hand.landmark):
                # The raw landmarks are normalized decimals
                # We multiply by the window dimensions to get exact pixel locations
                # on screen
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Append the formatted data
                self.lm_list.append([id, cx, cy])
                
        return self.lm_list