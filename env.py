import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

class XRayAttentionEnv(gym.Env):    # Custom RL Environment for Medical Image Glimpsing
    
    def __init__(self, data_list, crop_size=(84, 84), max_steps=10):
        super(XRayAttentionEnv, self).__init__()
        
        self.data_list = data_list
        self.crop_size = crop_size
        self.max_steps = max_steps
        
        # Actions: 0:Up, 1:Down, 2:Left, 3:Right, 4:Zoom, 5:Diagnose Pos, 6:Diagnose Neg
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=255, shape=(crop_size[0], crop_size[1], 1), dtype=np.uint8)
        
        self.current_idx = 0
        self.image = None
        self.label = None
        self.x = 0
        self.y = 0
        self.step_count = 0
        
    def reset(self, seed=None, options=None): # Resets env for the next episode/image
        super().reset(seed=seed)
        
        # Load next image
        img_path, self.label = self.data_list[self.current_idx]
        self.current_img_path = img_path #Save the path so we can plot it later
        self.current_idx = (self.current_idx + 1) % len(self.data_list)
        
        # Load grayscale image and resize to a base resolution (e.g., 224x224)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224), dtype=np.uint8) # Fallback if file missing
        else:
            img = cv2.resize(img, (224, 224))
            
        self.image = img
        self.x = 224 // 2   # Start in center
        self.y = 224 // 2
        self.step_count = 0
        
        return self._get_glimpse(), {}
        
    def _get_glimpse(self): # Extracts the 84x84 crop based on current coordinates
        half_w = self.crop_size[0] // 2
        
        # Ensure boundaries
        x_min = max(0, self.x - half_w)
        x_max = min(self.image.shape[1], self.x + half_w)
        y_min = max(0, self.y - half_w)
        y_max = min(self.image.shape[0], self.y + half_w)
        
        crop = self.image[y_min:y_max, x_min:x_max]
        crop = cv2.resize(crop, self.crop_size) # Force size if at edges
        return np.expand_dims(crop, axis=-1)
        
    def step(self, action): # Executes agent action and calculates reward
        self.step_count += 1
        done = False
        reward = -0.01 # Small negative reward to penalize taking too many steps
        info = {}
        
        stride = 20 # Pixels to move per step
        
        if action == 0: self.y -= stride
        elif action == 1: self.y += stride
        elif action == 2: self.x -= stride
        elif action == 3: self.x += stride
        elif action == 4: pass # Zoom logic can be implemented here by changing crop_size dynamically
        
        # Diagnosis Actions
        elif action in [5, 6]: 
            if self.step_count < 3:
                # Force the agent to look around. Penalty for guessing too early -> Fixes sitting issue very early on...
                reward = -0.5 
                # Done is NOT set to True here, so the episode continues -> Issue otherwise
            else:
                done = True
                if action == 5: # Diagnose Positive
                    reward = 5.0 if self.label == 1 else -1.0
                    info['diagnosis'] = 1 
                elif action == 6: # Diagnose Negative
                    # Increased TN reward so moving is mathematically worth it -> Somewhat fixes issue
                    reward = 1.0 if self.label == 0 else -5.0
                    info['diagnosis'] = 0
            
        if self.step_count >= self.max_steps: # Truncate if taking too long
            done = True
            reward = -1.0
            if 'diagnosis' not in info:
                info['diagnosis'] = -1 # Flag if it timed out without guessing
            
        self.x = np.clip(self.x, 0, 224)
        self.y = np.clip(self.y, 0, 224)
        
        info['x'] = int(self.x) # Track X coordinate
        info['y'] = int(self.y) # Track Y coordinate
        
        return self._get_glimpse(), reward, done, False, info # Return info dict
