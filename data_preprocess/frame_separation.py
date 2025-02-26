import cv2 
import os

class frameSeparation():
    def __init__(self):
        pass
    def extract_frames(self,video_dir, output_folder, frame_rate=2):
        """
        Extract frames from a video and save them as images.
        
        :param video_path: Path to the video file
        :param output_folder: Directory to save extracted frames
        :param frame_rate: Number of frames to extract per second
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        video_paths = os.listdir(video_dir)

        for idx,  video_path in enumerate(video_paths):
            video_path = video_dir+'/'+video_path        
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_interval = max(1, fps // frame_rate)  # Ensure at least one frame is extracted
        
            count = 0
            frame_number = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
            
                if count % frame_interval == 0:
                    frame_filename = os.path.join(output_folder, f"frame_{idx}_{frame_number:04d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    print(f"Saved {frame_filename}")
                    frame_number += 1
            
                count += 1
        
            cap.release()
            print("Frame extraction complete.")



if __name__=="__main__":

    # Example usage
    frame_sep_cls = frameSeparation()
    video_dir = "/home/sourav/workplace/leaf_disease_detection/dataset"  # Change to your video file path
    output_folder = "frames"  # Directory to save frames
    frame_rate = 1  # Extract 1 frame per second
    frame_sep_cls.extract_frames(video_dir, output_folder, frame_rate)
