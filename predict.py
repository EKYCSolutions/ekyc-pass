from tracker.tracker import TrackerBytetrack

def predict(args):
    tracker = TrackerBytetrack(args)
    tracker.track(show=True)

    
if __name__ == "__main__":
    # more config in file "tracker_config.py"
    args = {
        'path': "video_demo/volleyball.MOV",
    }

    predict(args)
