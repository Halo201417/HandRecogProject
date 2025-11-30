import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.5, track_con=0.5):
        """
        Inicialitation of the model for the hands of Mediapipe
        """
        
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4,8,12,16,20]   #Thumb, index, middle, ring, little fingers
        
    def find_hands(self, img, draw=True):
        """
        Processing the image and drawing the "skeleton"
        """
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    
        return img
    
    def find_position(self, img, hand_no=0):
        """
        Array of coordenates
        """
        
        self.lm_list = []
        
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            h, w, c = img.shape
            
            for id, lm in enumerate(my_hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
                
        return self.lm_list
    
    def count_fingers(self):
        """
        We count the fingers that are up
        """
        
        fingers = []
        
        if len(self.lm_list) != 0:
            #Checking if the point is at the right
            if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
                
            #The other fingers we compare the middle articulation
            for id in range(1, 5):
                if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
        return fingers